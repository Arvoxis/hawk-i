import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncpg
import logging
from datetime import datetime as _datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "hawki_db"),
    "user":     os.getenv("DB_USER", "hawki_user"),
    "password": os.getenv("DB_PASSWORD", "hawki"),
}

# ── Session identity ──────────────────────────────────────────
# Each process start gets its own timestamp-based table so sessions
# are isolated and historical data is never overwritten.
_SESSION_ID  = _datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_TABLE = f"detections_{_SESSION_ID}"

# Global connection pool
pool = None


async def init_db():
    global pool
    pool = await asyncpg.create_pool(**DB_CONFIG, statement_cache_size=0)

    async with pool.acquire() as conn:
        # Ensure PostGIS is available (no-op if already present)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

        # Create this session's table — flat lat/lon columns, no PostGIS
        # geography column, so the table works even without PostGIS DDL quirks.
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {CURRENT_TABLE} (
                id          SERIAL PRIMARY KEY,
                class_name  TEXT        NOT NULL,
                confidence  REAL        NOT NULL,
                severity    TEXT        NOT NULL,
                area_cm2    REAL        DEFAULT 0.0,
                lat         DOUBLE PRECISION DEFAULT 0.0,
                lon         DOUBLE PRECISION DEFAULT 0.0,
                altitude_m  REAL        DEFAULT 0.0,
                sam_score   REAL        DEFAULT 0.0,
                llm_report  TEXT,
                detected_at TIMESTAMPTZ DEFAULT NOW()
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

    logger.info(f"Session table created: {CURRENT_TABLE}")
    print(f"✓ Database connected  |  session table: {CURRENT_TABLE}")


async def save_detection(class_name, confidence, severity, area_cm2, lat, lon, altitude_m):
    """Insert a detection row into the current session table and return its id."""
    async with pool.acquire() as conn:
        row_id = await conn.fetchval(f"""
            INSERT INTO {CURRENT_TABLE}
                (class_name, confidence, severity, area_cm2, lat, lon, altitude_m)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        """, class_name, confidence, severity, area_cm2, lat, lon, altitude_m)
    logger.debug(f"Saved to DB: {class_name} | {severity}")
    return row_id


async def update_detection_sam(det_id: int, area_px: int, area_cm2: float, sam_score: float):
    """Back-fill SAM2 results for an already-saved detection row."""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            UPDATE {CURRENT_TABLE}
               SET area_cm2  = $2,
                   sam_score = $3
             WHERE id = $1
        """, det_id, area_cm2, sam_score)
    logger.debug(f"SAM2 updated row {det_id}: {area_cm2:.1f} cm²")


async def update_detection_report(det_id: int, report: str):
    """Save the LLM-generated inspection report for a detection row."""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            UPDATE {CURRENT_TABLE}
               SET llm_report = $2
             WHERE id = $1
        """, det_id, report)
    logger.debug(f"LLM report saved for row {det_id}")


async def get_detection_report(det_id: int) -> dict | None:
    """Return id + llm_report for a single detection row."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"""
            SELECT id, class_name, llm_report
              FROM {CURRENT_TABLE}
             WHERE id = $1
        """, det_id)
        return dict(row) if row else None


async def get_latest_detections(limit: int = 20) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT
                id, class_name, confidence, severity,
                area_cm2, lat, lon, altitude_m,
                sam_score, llm_report, detected_at
            FROM {CURRENT_TABLE}
            ORDER BY detected_at DESC
            LIMIT $1
        """, limit)
        return [dict(r) for r in rows]


async def get_filtered_detections(
    limit: int = 100,
    class_names: list | None = None,
    severities: list | None = None,
) -> list[dict]:
    """Return detections with optional filters on class_name and severity."""
    conditions: list[str] = []
    params: list = []

    if class_names:
        params.append(class_names)
        conditions.append(f"class_name = ANY(${len(params)})")
    if severities:
        params.append(severities)
        conditions.append(f"severity = ANY(${len(params)})")

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT
                id, class_name, confidence, severity,
                area_cm2, lat, lon, altitude_m,
                sam_score, llm_report, detected_at
            FROM {CURRENT_TABLE}
            {where_clause}
            ORDER BY detected_at DESC
            LIMIT ${len(params)}
        """, *params)
        return [dict(r) for r in rows]
