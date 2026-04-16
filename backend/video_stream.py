"""
video_stream.py — In-memory frame store for the live MJPEG feed.

latest_frame_jpeg holds the most recently received JPEG bytes from the drone
WebSocket.  set_latest_frame() is called by the receiver on every incoming
frame — zero disk access on the hot path.  get_latest_frame() is called by
/video_feed and /frame/latest with the same lock for thread safety.

SAM-annotated frames are written to data/frames/ only for L2/L3 severity
by sam3_worker (inside its thread-pool thread); they are NOT part of the live
feed — they are served as static files via the /frames/ mount for detection
card thumbnails.
"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

FRAME_PATH = 'data/live_frame.jpg'

# ── Live in-memory frame (updated by drone WS receiver on every frame) ─────────
_latest_frame_jpeg: bytes | None = None
_latest_frame_lock  = threading.Lock()


def set_latest_frame(jpeg_bytes: bytes) -> None:
    """Store the latest raw drone JPEG in memory (called by WS receiver)."""
    global _latest_frame_jpeg
    with _latest_frame_lock:
        _latest_frame_jpeg = jpeg_bytes


def get_latest_frame() -> bytes | None:
    """Return the latest in-memory JPEG, or None if no frame has arrived yet."""
    with _latest_frame_lock:
        return _latest_frame_jpeg


# ── Startup cleanup ────────────────────────────────────────────────────────────
_FRAMES_DIR = Path("data/frames")


def clear_frame_cache() -> None:
    """Delete all JPEG files from data/frames/ at server startup.

    Prevents stale SAM-annotated frames from a previous run appearing in
    detection card thumbnails.  Also resets the in-memory frame so the
    dashboard never serves a frame from a previous session.
    """
    global _latest_frame_jpeg
    with _latest_frame_lock:
        _latest_frame_jpeg = None

    if not _FRAMES_DIR.exists():
        return
    for f in _FRAMES_DIR.glob("*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass


# ── Placeholder ────────────────────────────────────────────────────────────────

def make_placeholder_jpeg() -> bytes:
    """Render a 'Waiting for drone feed' JPEG for the MJPEG stream.

    Called when get_latest_frame() returns None so /video_feed never stalls.
    """
    frame    = np.zeros((360, 640, 3), dtype=np.uint8)
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


# ── MJPEG /video_feed endpoint ─────────────────────────────────────────────────

router = APIRouter()


async def mjpeg_generator():
    last_mtime = 0
    last_frame = None
    while True:
        try:
            mtime = os.path.getmtime(FRAME_PATH)
            if mtime != last_mtime:
                with open(FRAME_PATH, 'rb') as f:
                    last_frame = f.read()
                last_mtime = mtime
        except FileNotFoundError:
            pass
        if last_frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + last_frame +
                b'\r\n'
            )
        else:
            placeholder = make_placeholder_jpeg()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + placeholder +
                b'\r\n'
            )
        await asyncio.sleep(0.033)


@router.get('/video_feed')
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
