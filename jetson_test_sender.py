"""
jetson_test_sender.py  —  Hawk-I Jetson → Laptop connection stress test.
=========================================================================
Simulates the Jetson streaming fake detection payloads to the laptop backend.
No camera, no YOLO, no SAM2 required — pure WebSocket stream.

Copy this file to the Jetson and run:
    python jetson_test_sender.py --server <laptop-LAN-IP>
    python jetson_test_sender.py --server 172.18.237.148 --fps 3 --count 20

Arguments:
    --server   Laptop IP address on the local network (REQUIRED from Jetson)
    --port     Backend port (default: 8000)
    --fps      Target frames per second (default: 3)
    --count    Number of frames to send; 0 = run forever (default: 10)
    --verbose  Show a log line for every frame

Dependencies (same as jetson_client.py):
    pip install websockets

NO camera, NO ultralytics, NO numpy needed — runs on bare Python.
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
import urllib.request

try:
    import websockets
except ImportError:
    sys.exit(
        "ERROR: 'websockets' not installed.\n"
        "  pip install websockets\n"
    )

# ── Tiny valid JPEG (1×1 grey pixel) — no cv2 needed ────────────────────────
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAA"
    "AAAAAAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAA"
    "AAAA/9oADAMBAAIRAxEAPwCwABmX/9k="
)

# ── Simulated defect classes ──────────────────────────────────────────────────
_YOLO_CLASSES  = ["crack", "spall", "corrosion", "delamination", "efflorescence"]
_GDINO_PHRASES = ["rust stain", "surface crack", "concrete spalling", "missing fastener"]

# ── Simulated GPS track (circular path, Bangalore area) ──────────────────────
_BASE_LAT = 12.9716
_BASE_LON = 77.5946
_RADIUS   = 0.002   # ~200 m radius


def _gps_for_frame(frame_idx: int) -> dict:
    """Simulate a slow circular flight path."""
    angle = (frame_idx * 5) % 360
    rad   = math.radians(angle)
    return {
        "lat":   round(_BASE_LAT + _RADIUS * math.sin(rad), 7),
        "lon":   round(_BASE_LON + _RADIUS * math.cos(rad), 7),
        "alt_m": round(10.0 + 2.0 * math.sin(math.radians(frame_idx * 7)), 2),
    }


def _random_box():
    x1 = random.randint(50, 400)
    y1 = random.randint(50, 300)
    return [x1, y1, x1 + random.randint(40, 200), y1 + random.randint(40, 150)]


def build_frame_payload(frame_idx: int) -> dict:
    """Create a realistic fake payload without a real camera."""
    n_yolo  = random.randint(0, 3)
    n_gdino = random.randint(0, 2)

    yolo_dets = [
        {
            "class": random.choice(_YOLO_CLASSES),
            "conf":  round(random.uniform(0.45, 0.98), 3),
            "box":   _random_box(),
        }
        for _ in range(n_yolo)
    ]
    gdino_dets = [
        {
            "phrase": random.choice(_GDINO_PHRASES),
            "conf":   round(random.uniform(0.50, 0.95), 3),
            "box":    _random_box(),
        }
        for _ in range(n_gdino)
    ]

    return {
        "timestamp":        time.time(),
        "gps":              _gps_for_frame(frame_idx),
        "yolo_detections":  yolo_dets,
        "gdino_detections": gdino_dets,
        "frame_jpeg":       _TINY_JPEG_B64,      # tiny placeholder — no camera needed
    }


# ── HTTP health check (before connecting WS) ─────────────────────────────────
def http_health_check(server: str, port: int) -> bool:
    url = f"http://{server}:{port}/health"
    print(f"Pre-flight: HTTP GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            print(f"  ✓ Server up — HTTP {r.status}")
            return True
    except Exception as exc:
        print(f"  ✗ Server unreachable: {exc}")
        print()
        print("  Tips:")
        print(f"    • Is the backend running on {server}:{port}?")
        print(f"    • Command: uvicorn backend.main:app --host 0.0.0.0 --port {port}")
        print(f"    • Is port {port} open in the laptop's firewall?")
        print(f"    •   Windows: netsh advfirewall firewall add rule name='Hawk-I' dir=in action=allow protocol=TCP localport={port}")
        return False


# ── Main streaming loop ───────────────────────────────────────────────────────
async def stream(server: str, port: int, fps: int, count: int, verbose: bool):
    uri      = f"ws://{server}:{port}/ws/drone"
    interval = 1.0 / max(1, fps)
    limit    = count if count > 0 else None   # None = run forever

    print(f"\nStreaming to {uri}")
    print(f"  FPS    : {fps}  (interval {interval:.3f}s)")
    print(f"  Frames : {'∞  (Ctrl-C to stop)' if limit is None else limit}")
    print()

    sent  = 0
    start = time.monotonic()

    RECONNECT_DELAY = 3.0

    while True:
        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=8,
            ) as ws:
                print(f"  ✓ Connected to {uri}")

                while limit is None or sent < limit:
                    t0       = time.monotonic()
                    payload  = build_frame_payload(sent)
                    await ws.send(json.dumps(payload))
                    sent    += 1

                    if verbose or sent % 10 == 0:
                        elapsed = time.monotonic() - start
                        fps_now = sent / elapsed if elapsed > 0 else 0.0
                        dets    = (
                            len(payload["yolo_detections"]) +
                            len(payload["gdino_detections"])
                        )
                        print(
                            f"  Frame {sent:>4d} | "
                            f"GPS {payload['gps']['lat']:.5f},{payload['gps']['lon']:.5f} | "
                            f"Dets: {dets} | "
                            f"Avg FPS: {fps_now:.1f}"
                        )

                    elapsed_frame = time.monotonic() - t0
                    await asyncio.sleep(max(0.0, interval - elapsed_frame))

                # Done
                break

        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            if limit is not None and sent >= limit:
                break
            print(f"  Connection lost: {exc}. Retrying in {RECONNECT_DELAY}s…")
            await asyncio.sleep(RECONNECT_DELAY)

    total_time = time.monotonic() - start
    avg_fps    = sent / total_time if total_time > 0 else 0.0
    print()
    print("=" * 60)
    print(f"  Done! Sent {sent} frames in {total_time:.1f}s  (avg {avg_fps:.1f} FPS)")
    print(f"  Check the laptop backend logs / /api/gcs/status endpoint.")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Hawk-I Jetson→Laptop connection stress test (no camera needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--server", default="172.18.239.242",
                   help="Laptop IP on LAN (e.g. 192.168.1.100). Default: 172.18.239.242")
    p.add_argument("--port",   default=8000, type=int,
                   help="Backend port (default: 8000)")
    p.add_argument("--fps",    default=3, type=int,
                   help="Target frames per second (default: 3)")
    p.add_argument("--count",  default=10, type=int,
                   help="Number of frames to send; 0 = unlimited (default: 10)")
    p.add_argument("--verbose", action="store_true",
                   help="Log every frame (not just every 10th)")
    args = p.parse_args()

    print("=" * 60)
    print("  Hawk-I  Jetson Test Sender")
    print(f"  Target : {args.server}:{args.port}")
    print("=" * 60)

    if not http_health_check(args.server, args.port):
        sys.exit(1)

    asyncio.run(stream(args.server, args.port, args.fps, args.count, args.verbose))


if __name__ == "__main__":
    main()
