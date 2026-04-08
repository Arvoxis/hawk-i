"""
jetson_client.py — Hawk-I Jetson-side capture + inference + streaming client.

Connects to the Hawk-I backend over WebSocket, captures frames from an
attached camera, runs YOLO inference, and streams detection payloads at the
configured FPS.  Reconnects automatically if the WebSocket drops.

Usage:
    python jetson_client.py
    python jetson_client.py --server 192.168.1.50
    python jetson_client.py --server 192.168.1.50 --camera 0 --model models/hawki_yolo11n.pt --fps 3
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import time

import cv2
import websockets
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jetson_client")

# Hardcoded GPS while real GPS module is not wired up.
PLACEHOLDER_GPS = {
    "lat": 12.9716,
    "lon": 77.5946,
    "alt_m": 12.0,
}

RECONNECT_DELAY_S = 3.0  # seconds to wait before retrying a dropped connection


# ── helpers ────────────────────────────────────────────────────────────────


def frame_to_jpeg_b64(frame_bgr) -> str:
    """Encode a BGR OpenCV frame as a JPEG base64 string."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def run_yolo(model: YOLO, frame_bgr) -> list[dict]:
    """Run YOLO on a BGR frame; return detections in Hawk-I schema."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        detections.append(
            {
                "class": class_name,
                "conf": round(conf, 4),
                "box": [round(x1), round(y1), round(x2), round(y2)],
            }
        )
    return detections


def build_payload(frame_bgr, yolo_detections: list[dict]) -> dict:
    return {
        "timestamp": time.time(),
        "gps": PLACEHOLDER_GPS,
        "yolo_detections": yolo_detections,
        "gdino_detections": [],   # GroundingDINO not run on Jetson
        "frame_jpeg": frame_to_jpeg_b64(frame_bgr),
    }


# ── main streaming loop ────────────────────────────────────────────────────


async def stream(server: str, camera: int, model_path: str, fps: int):
    uri = f"ws://{server}:8000/ws/drone"
    interval = 1.0 / max(1, fps)

    log.info("Loading YOLO model from %s …", model_path)
    model = YOLO(model_path)

    log.info("Opening camera %d …", camera)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}")

    log.info("Target: %s  FPS: %d  interval: %.3fs", uri, fps, interval)

    try:
        while True:
            log.info("Connecting to %s …", uri)
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    log.info("Connected.")
                    while True:
                        t_start = time.monotonic()

                        ok, frame = cap.read()
                        if not ok:
                            log.warning("Camera read failed — skipping frame")
                            await asyncio.sleep(interval)
                            continue

                        # Run YOLO in a thread so the event loop stays responsive.
                        loop = asyncio.get_running_loop()
                        detections = await loop.run_in_executor(
                            None, run_yolo, model, frame
                        )

                        payload = build_payload(frame, detections)
                        await ws.send(json.dumps(payload))

                        det_summary = ", ".join(
                            f"{d['class']}({d['conf']:.2f})" for d in detections
                        ) or "none"
                        log.info("Sent frame | detections: %s", det_summary)

                        elapsed = time.monotonic() - t_start
                        sleep_for = max(0.0, interval - elapsed)
                        await asyncio.sleep(sleep_for)

            except (websockets.ConnectionClosed, OSError) as exc:
                log.warning("Connection lost: %s — retrying in %.1fs …", exc, RECONNECT_DELAY_S)
                await asyncio.sleep(RECONNECT_DELAY_S)

    finally:
        cap.release()
        log.info("Camera released.")


# ── entry point ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Hawk-I Jetson streaming client")
    parser.add_argument(
        "--server",
        default="localhost",
        help="Backend host IP or hostname (default: localhost)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--model",
        default="models/hawki_yolo11n.pt",
        help="Path to YOLO weights file (default: models/hawki_yolo11n.pt)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Target frames per second to stream (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(stream(args.server, args.camera, args.model, args.fps))


if __name__ == "__main__":
    main()
