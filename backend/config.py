"""
config.py — Single source of truth for every Hawk-I GCS setting.

Every tunable in the ground-station pipeline is resolved here, once, from the
environment (``.env`` at the repo root).  Modules import the constants they
need instead of re-reading ``os.getenv`` or hardcoding literals.

Before this module existed the camera intrinsics were declared in three
different files with three different values, which made the cm2 figures
irreproducible.  Keep it that way: add new settings here, not inline.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# -- Model weights ------------------------------------------------------------
# Default location is models/ at the repo root.  Override the whole directory
# with HAWKI_MODELS_DIR, or an individual checkpoint with its own variable.
MODELS_DIR = Path(os.getenv("HAWKI_MODELS_DIR", PROJECT_ROOT / "models"))


def resolve_weights(filename: str, env_key: str | None = None) -> Path:
    """Locate a checkpoint, preferring an explicit override.

    Search order: ``$env_key`` -> ``MODELS_DIR/filename`` -> ``PROJECT_ROOT/filename``.
    The final candidate is returned even when nothing exists, so callers can
    log a path that tells the operator where the file was expected.
    """
    if env_key:
        override = os.getenv(env_key)
        if override:
            return Path(override)

    candidates = [MODELS_DIR / filename, PROJECT_ROOT / filename]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


YOLO_WEIGHTS    = resolve_weights("hawki_yolo11n.pt", "HAWKI_YOLO_WEIGHTS")
SAM2_CHECKPOINT = resolve_weights("sam2.1_hiera_small.pt", "HAWKI_SAM2_CHECKPOINT")
SAM2_MODEL_CFG  = os.getenv("HAWKI_SAM2_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")


# -- Camera intrinsics (drives the GSD -> cm2 conversion) ---------------------
# Defaults describe the Raspberry Pi HQ camera (IMX477) flown on the prototype.
CAMERA_SENSOR_WIDTH_MM = _env_float("CAMERA_SENSOR_WIDTH_MM", 6.287)
CAMERA_FOCAL_MM        = _env_float("CAMERA_FOCAL_MM", 4.74)
# Nominal capture width. Runtime code passes the *decoded* frame width instead;
# this value is only a fallback for callers that have no frame in hand.
CAMERA_IMAGE_WIDTH_PX  = _env_int("CAMERA_IMAGE_WIDTH_PX", 1920)

# Altitudes below this are treated as a bad GPS fix and replaced by DEFAULT_ALT_M.
MIN_ALT_M     = _env_float("MIN_ALT_M", 2.0)
DEFAULT_ALT_M = _env_float("DEFAULT_ALT_M", 10.0)


def gsd_cm_per_px(alt_m: float, image_width_px: int = CAMERA_IMAGE_WIDTH_PX) -> float:
    """Ground Sampling Distance in **cm per pixel** (pinhole camera model).

        GSD_m_per_px  = (alt_m * sensor_width_mm) / (focal_mm * image_width_px)
        GSD_cm_per_px = GSD_m_per_px * 100

    The millimetre units of sensor width and focal length cancel, so the raw
    ratio is already in metres; the *100 converts to centimetres.  Getting this
    factor wrong scales every reported defect area by its square, so the
    conversion lives here alone and is covered by tests/test_geometry.py.
    """
    if image_width_px <= 0:
        raise ValueError(f"image_width_px must be positive, got {image_width_px}")
    return (alt_m * CAMERA_SENSOR_WIDTH_MM) / (CAMERA_FOCAL_MM * image_width_px) * 100.0


def px_to_cm2(area_px: float, alt_m: float,
              image_width_px: int = CAMERA_IMAGE_WIDTH_PX) -> float:
    """Convert a pixel count to real-world area in cm2 at the given altitude."""
    gsd = gsd_cm_per_px(alt_m, image_width_px)
    return round(float(area_px) * (gsd ** 2), 2)


def resolve_altitude(alt_m) -> tuple[float, bool]:
    """Return ``(altitude, was_substituted)``.

    A null or implausibly low altitude means the GPS fix is unusable; the
    caller gets DEFAULT_ALT_M and a flag so it can warn exactly once.
    """
    try:
        alt = float(alt_m)
    except (TypeError, ValueError):
        return DEFAULT_ALT_M, True
    if alt < MIN_ALT_M:
        return DEFAULT_ALT_M, True
    return alt, False


# -- Severity thresholds (area-based, IRC-calibrated) -------------------------
SEVERITY_L3_CM2 = _env_float("SEVERITY_L3_CM2", 500.0)
SEVERITY_L2_CM2 = _env_float("SEVERITY_L2_CM2", 100.0)


def classify_severity(area_cm2: float) -> str:
    """Map a measured defect area to L1 (minor) / L2 (moderate) / L3 (critical)."""
    if area_cm2 >= SEVERITY_L3_CM2:
        return "L3"
    if area_cm2 >= SEVERITY_L2_CM2:
        return "L2"
    return "L1"


# -- SAM 2 --------------------------------------------------------------------
# Below this predicted-IoU score the mask is distrusted and the bounding-box
# pixel count is used for the area instead.
SAM_MASK_QUALITY_THRESHOLD = _env_float("SAM_MASK_QUALITY_THRESHOLD", 0.75)
# A tiny region SAM is also unsure about is almost always texture, not a defect.
SAM_FP_SCORE_MAX = _env_float("SAM_FP_SCORE_MAX", 0.25)
SAM_FP_AREA_MAX  = _env_float("SAM_FP_AREA_MAX", 15.0)


# -- DINOv2 -------------------------------------------------------------------
DINOV2_LOW_SIMILARITY_THRESHOLD = _env_float("DINOV2_LOW_SIMILARITY_THRESHOLD", 0.45)
DINOV2_PEER_FP_THRESHOLD        = _env_float("DINOV2_PEER_FP_THRESHOLD", 0.20)
DINOV2_MIN_CLASS_EXAMPLES       = _env_int("DINOV2_MIN_CLASS_EXAMPLES", 5)


# -- Detection intake ---------------------------------------------------------
# Minimum detector confidence to enter the processing queue at all.
MIN_DETECTION_CONF = _env_float("MIN_DETECTION_CONF", 0.45)
# Minimum confidence for a detection to earn an LLM-written report.
LLM_CONF_THRESHOLD = _env_float("LLM_CONF_THRESHOLD", 0.60)


# -- Ollama / LLM -------------------------------------------------------------
OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT         = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL     = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL     = f"{OLLAMA_BASE_URL}/api/tags"

LLM_MODEL      = os.getenv("LLM_MODEL", "gemma3:4b")
LLM_TIMEOUT_S  = _env_float("LLM_TIMEOUT_S", 30.0)
LLM_INTERVAL_S = _env_int("LLM_BATCH_INTERVAL", 30)

# Layers to offload to the GPU.  -1 means "let Ollama decide" (the normal case).
# Set OLLAMA_NUM_GPU=0 to force CPU inference -- required on hosts whose NVIDIA
# driver is older than the CUDA kernels Ollama ships, where GPU inference dies
# with "device kernel image is invalid".
OLLAMA_NUM_GPU = _env_int("OLLAMA_NUM_GPU", -1)

# Circuit breaker: after this many consecutive failures the LLM is considered
# down and calls short-circuit to the rule-based fallback for COOLDOWN_S
# seconds.  Without this, an unreachable Ollama costs LLM_TIMEOUT_S per
# detection and serialises the whole processing pipeline.
LLM_BREAKER_THRESHOLD  = _env_int("LLM_BREAKER_THRESHOLD", 3)
LLM_BREAKER_COOLDOWN_S = _env_float("LLM_BREAKER_COOLDOWN_S", 120.0)


def ollama_options(**overrides) -> dict:
    """Build the Ollama ``options`` block, applying the num_gpu override."""
    opts = {"temperature": 0.2, "num_predict": 256}
    opts.update(overrides)
    if OLLAMA_NUM_GPU >= 0:
        opts["num_gpu"] = OLLAMA_NUM_GPU
    return opts


# -- Ground-station YOLO (optional second detector on the GCS) ----------------
# Off by default: the Jetson already runs the detector, and re-running it on the
# ground doubles GPU load for little gain.  Enable when flying a camera-only
# payload that streams frames without on-board inference.
GS_YOLO_ENABLED = _env_bool("GS_YOLO_ENABLED", False)


# -- Storage paths ------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
FRAMES_DIR    = DATA_DIR / "frames"
REPORTS_DIR   = PROJECT_ROOT / "reports"
CAPTURES_DIR  = Path(os.getenv("GCS_SAVE_DIR", PROJECT_ROOT / "captures"))
DETECTION_LOG = Path(os.getenv("GCS_LOG_FILE", PROJECT_ROOT / "detections.jsonl"))


# -- GCS display --------------------------------------------------------------
GCS_HEADLESS       = _env_bool("GCS_HEADLESS", True)
GCS_AUTO_SAVE      = _env_bool("GCS_AUTO_SAVE", False)
GCS_AUTO_SAVE_CONF = _env_float("GCS_AUTO_SAVE_CONF", 0.70)


# -- Database -----------------------------------------------------------------
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     _env_int("DB_PORT", 5432),
    "database": os.getenv("DB_NAME", "hawki_db"),
    "user":     os.getenv("DB_USER", "hawki_user"),
    "password": os.getenv("DB_PASSWORD", "hawki"),
}


# -- Backend address (used by the dashboard and edge clients) -----------------
GCS_HOST    = os.getenv("GCS_HOST", "localhost")
GCS_PORT    = _env_int("GCS_PORT", 8000)
BACKEND_URL = f"http://{GCS_HOST}:{GCS_PORT}"


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Backend URL  : {BACKEND_URL}")
    print(f"YOLO weights : {YOLO_WEIGHTS}  (exists={YOLO_WEIGHTS.exists()})")
    print(f"SAM2 weights : {SAM2_CHECKPOINT}  (exists={SAM2_CHECKPOINT.exists()})")
    print(f"Ollama       : {OLLAMA_BASE_URL}  model={LLM_MODEL}  num_gpu={OLLAMA_NUM_GPU}")
    print(f"Intrinsics   : sensor={CAMERA_SENSOR_WIDTH_MM}mm focal={CAMERA_FOCAL_MM}mm")
    print(f"GSD @10m/1920px : {gsd_cm_per_px(10.0, 1920):.4f} cm/px")
    print(f"Severity     : L2>={SEVERITY_L2_CM2}cm2  L3>={SEVERITY_L3_CM2}cm2")
    print(f"Database     : {DB_CONFIG['user']}@{DB_CONFIG['host']}:"
          f"{DB_CONFIG['port']}/{DB_CONFIG['database']}")
