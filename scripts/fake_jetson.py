#!/usr/bin/env python3
"""
fake_jetson.py — Simulate a Jetson drone sending detection payloads
to the GCS backend WebSocket at ws://<GCS_HOST>:<GCS_PORT>/ws/drone (from .env).

Detection backend
-----------------
If a YOLO-World checkpoint is found (default: yolov8s-worldv2.pt in the repo
root), inference is performed with MultiQueryYOLOWorld on the loaded test frame.
Otherwise the script falls back to synthetic random detections so it can run
without a GPU or model file.

Backend command protocol (unchanged):
  {"type": "query", "query": <text>, "classes": [...]}  → update active filter
  {"type": "ping"}                                       → reply with {"type": "pong"}
"""

import asyncio
import base64
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websockets

# Allow importing multi_query_yoloworld from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load .env from repo root — single source of truth for all IPs/ports
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

_GCS_HOST = os.getenv("GCS_HOST", "172.18.239.242")
_GCS_PORT = os.getenv("GCS_PORT", "8000")
WS_URL    = f"ws://{_GCS_HOST}:{_GCS_PORT}/ws/drone"
INTERVAL = 3.0  # seconds between frames

# Fallback synthetic defect classes (used when no model is available)
_FALLBACK_CLASSES = [
    "Crack", "Spalling", "RustStain",
    "Exposed_reinforcement", "Scaling", "Efflorescence",
]

_BOXES = [
    [420, 310, 680, 490],
    [200, 150, 400, 320],
    [800, 500, 1100, 720],
    [100, 600, 350, 800],
    [1300, 200, 1600, 450],
    [600, 100, 900, 300],
    [50,   50, 250, 200],
]

_BASE_CONF = {
    "Crack":                 0.87,
    "Spalling":              0.72,
    "RustStain":             0.78,
    "Exposed_reinforcement": 0.91,
    "Scaling":               0.65,
    "Efflorescence":         0.70,
}

# Base GPS — Bengaluru area
BASE_LAT = 12.9716
BASE_LON = 77.5946
BASE_ALT = 12.5


# ── Model loading ─────────────────────────────────────────────────────────────

def _init_detector():
    """Try to load MultiQueryYOLOWorld.  Returns the detector or None."""
    model_path = Path(__file__).resolve().parent.parent / "yolov8s-worldv2.pt"
    if not model_path.exists():
        print(
            f"[fake_jetson] Model not found at {model_path}\n"
            "              Using synthetic detections — download yolov8s-worldv2.pt "
            "to enable real inference."
        )
        return None
    try:
        from multi_query_yoloworld import MultiQueryYOLOWorld
        detector = MultiQueryYOLOWorld(str(model_path), conf=0.25, iou_nms=0.4)
        print("[fake_jetson] MultiQueryYOLOWorld loaded — using real inference")
        return detector
    except Exception as exc:
        print(f"[fake_jetson] Could not load detector ({exc}) — falling back to synthetic")
        return None


# ── Frame loading ─────────────────────────────────────────────────────────────

def _load_or_create_frame() -> bytes:
    """Load a JPEG from data/test_frames/ or create a synthetic fallback."""
    frames_dir = Path("data/test_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        hits = list(frames_dir.glob(pattern))
        if hits:
            print(f"  [fake_jetson] Using existing frame: {hits[0]}")
            return hits[0].read_bytes()

    dest = frames_dir / "sample_crack.jpg"
    try:
        url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "a/a4/2007_06_03_Crack_in_Asphalt.jpg/"
            "320px-2007_06_03_Crack_in_Asphalt.jpg"
        )
        print("  [fake_jetson] Downloading sample frame …")
        urllib.request.urlretrieve(url, str(dest))
        print(f"  [fake_jetson] Saved: {dest}")
        return dest.read_bytes()
    except Exception as exc:
        print(f"  [fake_jetson] Download failed ({exc}), generating synthetic frame")

    rng = np.random.default_rng(42)
    frame = rng.integers(55, 120, (1080, 1920, 3), dtype=np.uint8)
    for _ in range(8):
        x1 = int(rng.integers(0, 1800))
        y1 = int(rng.integers(0, 900))
        x2 = int(np.clip(x1 + rng.integers(-400, 400), 0, 1919))
        y2 = int(np.clip(y1 + rng.integers(-400, 400), 0, 1079))
        cv2.line(frame, (x1, y1), (x2, y2), (18, 18, 18), 2)

    dest = frames_dir / "synthetic_frame.jpg"
    cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  [fake_jetson] Synthetic frame saved: {dest}")
    return dest.read_bytes()


# ── Detection helpers ─────────────────────────────────────────────────────────

def _make_synthetic(cycle: int, active_filter: str) -> list[dict]:
    """Fallback: return 1–3 randomised detections, filtered by active_filter."""
    if active_filter:
        words  = [w.strip().lower() for w in active_filter.replace(",", " ").split() if w.strip()]
        pool   = [c for c in _FALLBACK_CLASSES
                  if any(w in c.lower() or c.lower() in w for w in words)]
        pool   = pool or _FALLBACK_CLASSES
    else:
        pool = _FALLBACK_CLASSES

    n       = 1 if cycle % 2 == 0 else min(3, len(pool))
    classes = random.sample(pool, min(n, len(pool)))
    boxes   = random.sample(_BOXES, len(classes))

    dets = []
    for cls, box in zip(classes, boxes):
        conf = round(_BASE_CONF.get(cls, 0.75) + random.uniform(-0.05, 0.05), 3)
        conf = max(0.50, min(0.99, conf))
        dets.append({"class": cls, "conf": conf, "box": box})
    return dets


def _run_inference(
    detector,
    frame_bgr: np.ndarray,
    active_filter: str,
) -> list[dict]:
    """Run MultiQueryYOLOWorld inference and apply active_filter post-hoc.

    Converts Detection objects to the wire format expected by the backend:
        {"class": str, "conf": float, "box": [x1, y1, x2, y2]}
    """
    detections = detector.predict(frame_bgr)

    # Apply class filter from the last backend query (substring match on canonical label)
    if active_filter:
        words      = [w.strip().lower() for w in active_filter.replace(",", " ").split() if w.strip()]
        detections = [
            d for d in detections
            if any(w in d.label.lower() or d.label.lower() in w for w in words)
        ] or detections  # fall back to all if filter matches nothing

    return [
        {"class": d.label, "conf": round(d.confidence, 3), "box": d.bbox}
        for d in detections
    ]


# ── Main WebSocket loop ───────────────────────────────────────────────────────

async def stream_to_gcs():
    frame_bytes = _load_or_create_frame()
    frame_b64   = base64.b64encode(frame_bytes).decode()

    # Decode once for inference (BGR numpy array)
    buf       = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    detector = _init_detector()

    lat, lon, alt = BASE_LAT, BASE_LON, BASE_ALT
    cycle = 0

    print(f"\n[fake_jetson] Connecting to {WS_URL} …\n")

    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=10,
                ping_timeout=5,
            ) as ws:
                print(f"[fake_jetson] Connected ✓  (sending every {INTERVAL}s)")

                _filter = [""]  # _filter[0] holds the active query string

                async def _recv_backend():
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        if msg.get("type") == "query":
                            _filter[0] = msg.get("query", "")
                            print(f"[fake_jetson] Query received: {_filter[0]!r}")

                        elif msg.get("type") == "ping":
                            await ws.send(json.dumps({"type": "pong"}))
                            print("[fake_jetson] Ping received — pong sent")

                async def _send_loop():
                    nonlocal lat, lon, cycle
                    while True:
                        lat = round(lat + random.uniform(-0.0002, 0.0002), 6)
                        lon = round(lon + random.uniform(-0.0002, 0.0002), 6)

                        if detector is not None and frame_bgr is not None:
                            yolo_dets = _run_inference(detector, frame_bgr, _filter[0])
                        else:
                            yolo_dets = _make_synthetic(cycle, _filter[0])

                        payload = {
                            "timestamp":        time.time(),
                            "gps":              {"lat": lat, "lon": lon, "alt_m": alt},
                            "yolo_detections":  yolo_dets,
                            "gdino_detections": [],
                            "frame_jpeg":       frame_b64,
                        }

                        await ws.send(json.dumps(payload))

                        backend = "model" if detector else "synthetic"
                        det_summary = [
                            "{cls}@{c:.2f}".format(cls=d["class"], c=d["conf"])
                            for d in yolo_dets
                        ]
                        filter_tag = f"  [filter={_filter[0]!r}]" if _filter[0] else ""
                        print(
                            f"[fake_jetson] cycle={cycle:04d}  GPS=({lat:.4f},{lon:.4f})  "
                            f"[{backend}] dets={det_summary}{filter_tag}"
                        )

                        cycle += 1
                        await asyncio.sleep(INTERVAL)

                recv_task = asyncio.create_task(_recv_backend())
                send_task = asyncio.create_task(_send_loop())

                done, pending = await asyncio.wait(
                    [recv_task, send_task],
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                for t in pending:
                    t.cancel()
                for t in done:
                    if not t.cancelled():
                        exc = t.exception()
                        if exc:
                            raise exc

        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            print(f"[fake_jetson] Connection lost ({exc}), retrying in 3 s …")
            await asyncio.sleep(3)
        except KeyboardInterrupt:
            print("\n[fake_jetson] Stopped by user.")
            return


if __name__ == "__main__":
    asyncio.run(stream_to_gcs())
