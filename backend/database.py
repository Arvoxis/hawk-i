import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncpg
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "hawki_db"),
    "user":     os.getenv("DB_USER", "hawki_user"),
    "password": os.getenv("DB_PASSWORD", "hawki"),
}

# Global connection pool
pool = None

async def init_db():
    global pool
    pool = await asyncpg.create_pool(**DB_CONFIG)

    # Add SAM2 columns if they don't exist yet
    async with pool.acquire() as conn:
        await conn.execute("""
            ALTER TABLE detections
                ADD COLUMN IF NOT EXISTS area_px  INTEGER,
                ADD COLUMN IF NOT EXISTS sam_score REAL
        """)

    print("✓ Database connected")

async def save_detection(class_name, confidence, severity, area_cm2, lat, lon, altitude_m):
    """Insert a detection row and return its auto-generated id."""
    async with pool.acquire() as conn:
        row_id = await conn.fetchval("""
            INSERT INTO detections
                (class_name, confidence, severity, area_cm2, location, altitude_m)
            VALUES
                ($1, $2, $3, $4, ST_SetSRID(ST_MakePoint($5, $6), 4326), $7)
            RETURNING id
        """, class_name, confidence, severity, area_cm2, lon, lat, altitude_m)
    print(f"  ✓ Saved to DB: {class_name} | {severity}")
    return row_id

async def update_detection_sam(det_id: int, area_px: int, area_cm2: float, sam_score: float):
    """Back-fill SAM2 results for an already-saved detection row."""
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE detections
               SET area_px   = $2,
                   area_cm2  = $3,
                   sam_score = $4
             WHERE id = $1
        """, det_id, area_px, area_cm2, sam_score)
    print(f"  ✓ SAM2 updated row {det_id}: {area_cm2:.1f} cm²")

async def get_latest_detections(limit=20):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                class_name, confidence, severity, area_cm2,
                ST_Y(location::geometry) as lat,
                ST_X(location::geometry) as lon,
                altitude_m, detected_at
            FROM detections
            ORDER BY detected_at DESC
            LIMIT $1
        """, limit)
        return [dict(r) for r in rows]