import logging
from datetime import datetime as _datetime

import config

logger = logging.getLogger(__name__)

DB_CONFIG = config.DB_CONFIG

# ── Session identity ──────────────────────────────────────────
# Each process start gets its own timestamp-based table so sessions
# are isolated and historical data is never overwritten.
_SESSION_ID  = _datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_TABLE = f"detections_{_SESSION_ID}"

# Persistent across sessions, unlike CURRENT_TABLE.  This is what makes
# "has this defect been seen before, and has it grown?" answerable at all:
# session tables are per-process, so a lookup scoped to one could only ever
# find defects from the same flight.
EMBEDDING_TABLE = "defect_embeddings"

# Global connection pool
pool = None


async def init_db():
    # Imported here, not at module level: asyncpg is touched on this one line
    # only, and a top-level import makes `import database` -- and so the
    # pure-logic unit tests that pull in processing_worker -- require a
    # database driver they never call.
    import asyncpg

    global pool
    pool = await asyncpg.create_pool(**DB_CONFIG, statement_cache_size=0)

    async with pool.acquire() as conn:
        # Ensure PostGIS is available (no-op if already present)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

        # severity is NULL until the processing worker completes SAM2 + LLM.
        # The dashboard only shows rows WHERE severity IS NOT NULL (fully processed).
        # raw_box_json is consumed by the processing worker (no frame bytes stored).
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {CURRENT_TABLE} (
                id             SERIAL PRIMARY KEY,
                class_name     TEXT             NOT NULL,
                confidence     REAL             NOT NULL,
                severity       TEXT,
                area_cm2       REAL             DEFAULT 0.0,
                lat            DOUBLE PRECISION DEFAULT 0.0,
                lon            DOUBLE PRECISION DEFAULT 0.0,
                altitude_m     REAL             DEFAULT 0.0,
                sam_score      REAL             DEFAULT 0.0,
                image_path     TEXT,
                source_model   TEXT             DEFAULT 'yolo_world',
                llm_report     TEXT,
                raw_box_json   TEXT,
                detected_at    TIMESTAMPTZ      DEFAULT NOW()
            )
        """)

        # Optionally add a PostGIS point column for spatial queries.
        # Silently ignored if PostGIS isn't fully set up or column exists.
        try:
            await conn.execute(
                f"SELECT AddGeometryColumn('{CURRENT_TABLE}','geom',4326,'POINT',2)"
            )
        except Exception:
            pass

        # DINOv2 columns — added with IF NOT EXISTS so existing tables survive.
        for col_ddl in [
            f"ALTER TABLE {CURRENT_TABLE} ADD COLUMN IF NOT EXISTS embedding BYTEA",
            f"ALTER TABLE {CURRENT_TABLE} ADD COLUMN IF NOT EXISTS confidence_adjusted REAL",
            f"ALTER TABLE {CURRENT_TABLE} ADD COLUMN IF NOT EXISTS dinov2_flagged BOOLEAN DEFAULT FALSE",
            f"ALTER TABLE {CURRENT_TABLE} ADD COLUMN IF NOT EXISTS similar_ids TEXT",
        ]:
            try:
                await conn.execute(col_ddl)
            except Exception:
                pass

    await _init_embedding_archive()

    logger.info("Database connected | session table: %s", CURRENT_TABLE)


async def _init_embedding_archive() -> None:
    """Create the cross-session embedding archive if it does not exist."""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {EMBEDDING_TABLE} (
                id           BIGSERIAL PRIMARY KEY,
                session_id   TEXT             NOT NULL,
                detection_id INTEGER          NOT NULL,
                class_name   TEXT             NOT NULL,
                embedding    BYTEA            NOT NULL,
                area_cm2     REAL             DEFAULT 0.0,
                severity     TEXT,
                confidence   REAL             DEFAULT 0.0,
                lat          DOUBLE PRECISION DEFAULT 0.0,
                lon          DOUBLE PRECISION DEFAULT 0.0,
                sam_score    REAL             DEFAULT 0.0,
                altitude_m   REAL             DEFAULT 0.0,
                detected_at  TIMESTAMPTZ      DEFAULT NOW(),
                UNIQUE (session_id, detection_id)
            )
        """)
        # The archive outlives any one release, so CREATE TABLE IF NOT EXISTS
        # will not add columns introduced later.  sam_score separates a
        # SAM-measured prior area from a bbox estimate (comparing the two is
        # meaningless); altitude_m lets a past area be recomputed and audited.
        for _col in ("sam_score", "altitude_m"):
            await conn.execute(
                f"ALTER TABLE {EMBEDDING_TABLE} "
                f"ADD COLUMN IF NOT EXISTS {_col} REAL DEFAULT 0.0"
            )
        # Class is the only filter every lookup applies.
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {EMBEDDING_TABLE}_class_idx
                ON {EMBEDDING_TABLE} (class_name, detected_at DESC)
        """)
    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM {EMBEDDING_TABLE}")
        classes = await conn.fetchval(
            f"SELECT COUNT(DISTINCT class_name) FROM {EMBEDDING_TABLE}")
    logger.info(
        "Embedding archive ready: %d embedding(s) across %d class(es) from "
        "previous inspections", int(total or 0), int(classes or 0),
    )


async def archive_embedding(
    detection_id: int,
    class_name: str,
    embedding_bytes: bytes,
    area_cm2: float,
    severity: str | None,
    confidence: float,
    lat: float,
    lon: float,
    sam_score: float = 0.0,
    altitude_m: float = 0.0,
) -> None:
    """Copy one detection's embedding into the cross-session archive."""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            INSERT INTO {EMBEDDING_TABLE}
                (session_id, detection_id, class_name, embedding,
                 area_cm2, severity, confidence, lat, lon,
                 sam_score, altitude_m)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (session_id, detection_id) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    area_cm2   = EXCLUDED.area_cm2,
                    severity   = EXCLUDED.severity,
                    sam_score  = EXCLUDED.sam_score,
                    altitude_m = EXCLUDED.altitude_m
        """, _SESSION_ID, detection_id, class_name, embedding_bytes,
            area_cm2, severity, confidence, lat, lon,
            sam_score, altitude_m)


async def fetch_archived_embeddings(
    class_name: str,
    limit: int = 2000,
    exclude_current_session: bool = False,
) -> list[dict]:
    """Return archived embeddings for one defect class, newest first."""
    guard = " AND session_id <> $2" if exclude_current_session else ""
    params: list = [class_name]
    if exclude_current_session:
        params.append(_SESSION_ID)

    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT id, session_id, detection_id, embedding, area_cm2,
                   severity, lat, lon, sam_score, altitude_m, detected_at
              FROM {EMBEDDING_TABLE}
             WHERE class_name = $1{guard}
             ORDER BY detected_at DESC
             LIMIT {int(limit)}
        """, *params)
        return [dict(r) for r in rows]


async def archive_stats() -> dict:
    """Summary of the archive, for /health and diagnostics."""
    if pool is None:
        return {"available": False}
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT COUNT(*) AS total,
                       COUNT(DISTINCT class_name) AS classes,
                       COUNT(DISTINCT session_id) AS sessions
                  FROM {EMBEDDING_TABLE}
            """)
        return {
            "available": True,
            "embeddings": int(row["total"] or 0),
            "classes":    int(row["classes"] or 0),
            "sessions":   int(row["sessions"] or 0),
        }
    except Exception:
        return {"available": False}


async def save_detection_raw(
    class_name: str,
    confidence: float,
    lat: float,
    lon: float,
    altitude_m: float,
    source_model: str = "yolo_world",
    raw_box_json: str | None = None,
) -> int:
    """Insert a bare detection row — severity/area_cm2 stay NULL until the
    processing worker completes SAM2 and LLM.  Returns the new row id."""
    async with pool.acquire() as conn:
        row_id = await conn.fetchval(f"""
            INSERT INTO {CURRENT_TABLE}
                (class_name, confidence, lat, lon, altitude_m,
                 source_model, raw_box_json)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        """, class_name, confidence, lat, lon, altitude_m,
            source_model, raw_box_json)

        try:
            await conn.execute(f"""
                UPDATE {CURRENT_TABLE}
                   SET geom = ST_SetSRID(ST_MakePoint($2, $1), 4326)
                 WHERE id = $3
            """, lat, lon, row_id)
        except Exception:
            pass

    logger.debug("Queued raw detection: %s conf=%.2f id=%d", class_name, confidence, row_id)
    return row_id


async def save_detection(
    class_name, confidence, severity, area_cm2,
    lat, lon, altitude_m,
    source_model: str = "yolo_world",
    image_path: str | None = None,
):
    """Legacy full-insert (used by /api/segment). Sets severity immediately."""
    async with pool.acquire() as conn:
        row_id = await conn.fetchval(f"""
            INSERT INTO {CURRENT_TABLE}
                (class_name, confidence, severity, area_cm2,
                 lat, lon, altitude_m, source_model, image_path)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """, class_name, confidence, severity, area_cm2,
            lat, lon, altitude_m, source_model, image_path)

        try:
            await conn.execute(f"""
                UPDATE {CURRENT_TABLE}
                   SET geom = ST_SetSRID(ST_MakePoint($2, $1), 4326)
                 WHERE id = $3
            """, lat, lon, row_id)
        except Exception:
            pass

    logger.debug("Saved detection: %s | %s | src=%s", class_name, severity, source_model)
    return row_id



async def update_detection_sam(
    det_id: int,
    area_px: int,
    area_cm2: float,
    sam_score: float,
    image_path: str | None = None,
    severity: str | None = None,
):
    """Back-fill SAM2/SAM3 results for an already-saved detection row.

    ``severity`` is the area-based classification produced by sam2_worker;
    when provided it overwrites the initial confidence-based severity so the
    DB always reflects the real-world defect size.
    """
    async with pool.acquire() as conn:
        if severity:
            await conn.execute(f"""
                UPDATE {CURRENT_TABLE}
                   SET area_cm2   = $2,
                       sam_score  = $3,
                       image_path = COALESCE($4, image_path),
                       severity   = $5
                 WHERE id = $1
            """, det_id, area_cm2, sam_score, image_path, severity)
        else:
            await conn.execute(f"""
                UPDATE {CURRENT_TABLE}
                   SET area_cm2   = $2,
                       sam_score  = $3,
                       image_path = COALESCE($4, image_path)
                 WHERE id = $1
            """, det_id, area_cm2, sam_score, image_path)
    logger.debug(f"SAM3 updated row {det_id}: {area_cm2:.1f} cm² sev={severity or '—'}")


async def update_detection_report(
    det_id: int,
    report: str,
    only_if_empty: bool = False,
):
    """Save the LLM-generated inspection report for a detection row.

    ``only_if_empty`` is used by the periodic batch sweep so it can fill gaps
    without clobbering the richer per-detection report that the processing
    worker writes (that one carries DINOv2 and similar-defect context).
    """
    guard = " AND llm_report IS NULL" if only_if_empty else ""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            UPDATE {CURRENT_TABLE}
               SET llm_report = $2
             WHERE id = $1{guard}
        """, det_id, report)
    logger.debug(f"LLM report saved for row {det_id}")


async def count_missing_llm_reports() -> int:
    """Number of processed detections still waiting for a report."""
    async with pool.acquire() as conn:
        return int(await conn.fetchval(f"""
            SELECT COUNT(*) FROM {CURRENT_TABLE}
             WHERE severity IS NOT NULL AND llm_report IS NULL
        """) or 0)


async def get_detection_report(det_id: int) -> dict | None:
    """Return id + llm_report for a single detection row."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"""
            SELECT id, class_name, llm_report
              FROM {CURRENT_TABLE}
             WHERE id = $1
        """, det_id)
        return dict(row) if row else None


async def get_recent_llm_reports(seconds: int = 60, limit: int = 10) -> list[dict]:
    """Return detections that have an llm_report written in the last `seconds` seconds."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT id, class_name, confidence, severity,
                   area_cm2, lat, lon, altitude_m, llm_report, detected_at,
                   raw_box_json, sam_score, similar_ids
              FROM {CURRENT_TABLE}
             WHERE llm_report IS NOT NULL
               AND detected_at >= NOW() - INTERVAL '{seconds} seconds'
             ORDER BY detected_at DESC
             LIMIT $1
        """, limit)
        return [dict(r) for r in rows]


async def get_latest_detections(limit: int = 20) -> list[dict]:
    """Return only fully-processed detections (severity IS NOT NULL)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT
                id, class_name, confidence, severity,
                area_cm2, lat, lon, altitude_m,
                sam_score, image_path, source_model,
                llm_report, detected_at,
                raw_box_json, dinov2_flagged, similar_ids
            FROM {CURRENT_TABLE}
            WHERE severity IS NOT NULL
            ORDER BY detected_at DESC
            LIMIT $1
        """, limit)
        return [dict(r) for r in rows]


async def get_recent_detections_for_llm(
    seconds: int = 30,
    min_conf: float = 0.60,
    only_missing_report: bool = False,
) -> list[dict]:
    """Return recent detections above `min_conf` that are fully processed.

    ``only_missing_report`` restricts the result to rows with no llm_report,
    so the batch sweep never regenerates a report that already exists.
    """
    guard = " AND llm_report IS NULL" if only_missing_report else ""
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT id, class_name, confidence, severity, area_cm2, lat, lon, altitude_m
              FROM {CURRENT_TABLE}
             WHERE detected_at >= NOW() - INTERVAL '{seconds} seconds'
               AND confidence > $1
               AND severity IS NOT NULL{guard}
             ORDER BY detected_at DESC
        """, min_conf)
        return [dict(r) for r in rows]


async def update_detection_dinov2(
    det_id: int,
    embedding_bytes: bytes,
    confidence_adjusted: float,
    dinov2_flagged: bool,
    severity: str | None = None,
    similar_ids: str | None = None,
) -> None:
    """Save DINOv2 embedding, re-scored confidence, flag, and similar IDs.

    If severity is provided (downgraded after DINOv2 check), it overwrites
    the SAM2-based severity already stored for this row.
    """
    async with pool.acquire() as conn:
        if severity is not None:
            await conn.execute(
                f"""
                UPDATE {CURRENT_TABLE}
                   SET embedding           = $2,
                       confidence_adjusted = $3,
                       dinov2_flagged      = $4,
                       severity            = $5,
                       similar_ids         = COALESCE($6, similar_ids)
                 WHERE id = $1
                """,
                det_id, embedding_bytes, confidence_adjusted,
                dinov2_flagged, severity, similar_ids,
            )
        else:
            await conn.execute(
                f"""
                UPDATE {CURRENT_TABLE}
                   SET embedding           = $2,
                       confidence_adjusted = $3,
                       dinov2_flagged      = $4,
                       similar_ids         = COALESCE($5, similar_ids)
                 WHERE id = $1
                """,
                det_id, embedding_bytes, confidence_adjusted,
                dinov2_flagged, similar_ids,
            )
    logger.debug("DINOv2 saved for row %d: flagged=%s sev=%s", det_id, dinov2_flagged, severity)


async def get_detection_embedding(det_id: int) -> dict | None:
    """Return id, class_name, and raw BYTEA embedding for similarity lookups."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT id, class_name, embedding, lat, lon, area_cm2, detected_at
              FROM {CURRENT_TABLE}
             WHERE id = $1
            """,
            det_id,
        )
        return dict(row) if row else None


async def get_severity_counts() -> dict:
    """Return per-severity detection counts for the site health score.

    Bucketing happens in SQL.  The previous version grouped by the raw
    confidence column, which produced one group per distinct float value and
    made the query scale with the number of rows rather than the four buckets.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT
                COUNT(*)                                                       AS total,
                COUNT(*) FILTER (WHERE severity = 'L3' AND confidence > 0.85)  AS critical,
                COUNT(*) FILTER (WHERE severity = 'L3' AND confidence <= 0.85) AS high,
                COUNT(*) FILTER (WHERE severity = 'L2')                        AS medium,
                COUNT(*) FILTER (WHERE severity NOT IN ('L2', 'L3'))           AS low
              FROM {CURRENT_TABLE}
             WHERE severity IS NOT NULL
            """
        )
    return {k: int(row[k] or 0) for k in ("total", "critical", "high", "medium", "low")}


async def get_filtered_detections(
    limit: int = 100,
    class_names: list | None = None,
    severities: list | None = None,
) -> list[dict]:
    """Return fully-processed detections with optional filters."""
    # Always require severity IS NOT NULL (processing complete)
    conditions: list[str] = ["severity IS NOT NULL"]
    params: list = []

    if class_names:
        params.append(class_names)
        conditions.append(f"class_name = ANY(${len(params)})")
    if severities:
        params.append(severities)
        conditions.append(f"severity = ANY(${len(params)})")

    where_clause = "WHERE " + " AND ".join(conditions)
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT
                id, class_name, confidence, severity,
                area_cm2, lat, lon, altitude_m,
                sam_score, image_path, source_model,
                llm_report, detected_at,
                raw_box_json, dinov2_flagged, similar_ids
            FROM {CURRENT_TABLE}
            {where_clause}
            ORDER BY detected_at DESC
            LIMIT ${len(params)}
        """, *params)
        return [dict(r) for r in rows]
