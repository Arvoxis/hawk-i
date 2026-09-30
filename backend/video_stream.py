"""
video_stream.py — In-memory frame store for the live MJPEG feed.

The most recently received JPEG from the drone WebSocket is held in memory and
served by /video_feed and /frame/latest in main.py.  set_latest_frame() is
called on the hot path (the WS receiver) and again by the processing worker
once SAM 2 has drawn its mask overlay, so the operator sees the annotated frame
rather than the raw camera image whenever a detection is present.

There is deliberately no disk I/O here.  SAM-annotated stills for L2/L3
detections are written separately by sam2_worker into data/frames/ and served
as static files via the /frames mount — those are detection-card thumbnails,
not the live feed.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

import config

# ── Live in-memory frame (updated by the WS receiver and processing worker) ───
_latest_frame_jpeg: bytes | None = None
_latest_frame_lock = threading.Lock()


def set_latest_frame(jpeg_bytes: bytes) -> None:
    """Store the latest JPEG in memory."""
    global _latest_frame_jpeg
    with _latest_frame_lock:
        _latest_frame_jpeg = jpeg_bytes


def get_latest_frame() -> bytes | None:
    """Return the latest in-memory JPEG, or None if no frame has arrived yet."""
    with _latest_frame_lock:
        return _latest_frame_jpeg


# ── Startup cleanup ───────────────────────────────────────────────────────────
_FRAMES_DIR: Path = config.FRAMES_DIR


def clear_frame_cache() -> int:
    """Delete stale annotated JPEGs from data/frames/ at server startup.

    Prevents thumbnails from a previous mission leaking into this session's
    detection cards.  Also resets the in-memory frame so the dashboard never
    serves a frame captured before the current run.  Returns the file count.
    """
    global _latest_frame_jpeg
    with _latest_frame_lock:
        _latest_frame_jpeg = None

    if not _FRAMES_DIR.exists():
        return 0

    removed = 0
    for f in _FRAMES_DIR.glob("*.jpg"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# ── Placeholder ───────────────────────────────────────────────────────────────

def make_placeholder_jpeg() -> bytes:
    """Render a 'Waiting for drone feed' JPEG for the MJPEG stream.

    Served when get_latest_frame() returns None so /video_feed never stalls.
    """
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (18, 18, 28)

    cv2.putText(frame, "HAWK-I GCS",
                (215, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 2)
    cv2.putText(frame, "Waiting for drone feed...",
                (155, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 1)
    dots = "." * (int(time.time() * 2) % 4)
    cv2.putText(frame, dots,
                (475, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 1)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()
