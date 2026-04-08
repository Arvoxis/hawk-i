"""
video_stream.py — Thread-safe latest-frame store + MJPEG placeholder generator.

Usage in main.py:
    from video_stream import set_frame, get_frame, make_placeholder_jpeg
"""

import threading
import time

import cv2
import numpy as np

# ── Thread-safe frame store ────────────────────────────────────────────────────
_lock: threading.Lock          = threading.Lock()
_latest_frame_bytes: bytes | None = None   # raw JPEG bytes from the Jetson
_latest_frame_time:  float        = 0.0    # monotonic timestamp of last update

# Frames older than this are treated as stale (drone disconnected / paused).
_FRAME_TTL_S: float = 5.0


def set_frame(jpeg_bytes: bytes) -> None:
    """Store the latest decoded JPEG bytes. Called from the drone WS handler."""
    global _latest_frame_bytes, _latest_frame_time
    with _lock:
        _latest_frame_bytes = jpeg_bytes
        _latest_frame_time  = time.monotonic()


def get_frame() -> bytes | None:
    """
    Return the latest JPEG bytes, or None if nothing has been received yet
    or the last frame is older than _FRAME_TTL_S seconds.
    """
    with _lock:
        if _latest_frame_bytes is None:
            return None
        if time.monotonic() - _latest_frame_time > _FRAME_TTL_S:
            return None
        return _latest_frame_bytes


# ── Placeholder generator ──────────────────────────────────────────────────────

def make_placeholder_jpeg() -> bytes:
    """
    Render a 'Waiting for drone feed' frame as a JPEG.
    Called when get_frame() returns None so the MJPEG stream never stalls.
    """
    frame        = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:]     = (18, 18, 28)                        # near-black dark background

    cv2.putText(
        frame, "HAWK-I GCS",
        (215, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 2,
    )
    cv2.putText(
        frame, "Waiting for drone feed...",
        (155, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 1,
    )
    # Animated dots (changes every ~0.5 s)
    dots = "." * (int(time.time() * 2) % 4)
    cv2.putText(
        frame, dots,
        (475, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 1,
    )

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()
