"""
test_connection.py  —  Hawk-I backend connectivity test script.
No camera, no YOLO, no SAM2 required.

┌─ On the LAPTOP (self-test) ────────────────────────────────┐
│  python test_connection.py                                 │
│  python test_connection.py --server localhost              │
└────────────────────────────────────────────────────────────┘

┌─ On the JETSON (cross-device test) ────────────────────────┐
│  python test_connection.py --server <laptop-LAN-IP>        │
│  e.g.  python test_connection.py --server 192.168.1.100    │
└────────────────────────────────────────────────────────────┘

Two checks are performed:
  1. HTTP GET /health       — confirms TCP reachability
  2. WebSocket /ws/drone    — sends a fake detection payload
"""

import argparse
import asyncio
import json
import time
import urllib.request

import websockets  # pip install websockets


# ── Tiny valid JPEG (1×1 grey pixel, base64-encoded) ─────────────────────────
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAA"
    "AAAAAAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAA"
    "AAAA/9oADAMBAAIRAxEAPwCwABmX/9k="
)


def build_payload() -> dict:
    """Construct a realistic-looking but fake drone detection payload."""
    return {
        "timestamp": time.time(),
        "gps": {
            "lat": 12.9716,
            "lon": 77.5946,
            "alt_m": 10.0,
        },
        "yolo_detections": [
            {
                "class": "cracked concrete",
                "conf": 0.82,
                "box": [120, 150, 380, 420],
            }
        ],
        "gdino_detections": [
            {
                "phrase": "rust stain",
                "conf": 0.75,
                "box": [200, 100, 500, 350],
            }
        ],
        "frame_jpeg": _TINY_JPEG_B64,
    }


# ── Step 1: HTTP health check ─────────────────────────────────────────────────

def check_http_health(server: str) -> bool:
    url = f"http://{server}:8000/health"
    print(f"\n[1/2] HTTP health check → {url}")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode()
            print(f"      ✓  HTTP {resp.status}  — {body}")
            return True
    except Exception as exc:
        print(f"      ✗  FAILED: {exc}")
        print()
        print("      Possible causes:")
        print("        • Backend not running — start it with:")
        print("            cd hawki")
        print("            uvicorn backend.main:app --host 0.0.0.0 --port 8000")
        print("        • Wrong IP address (use laptop's LAN IP, not 127.0.0.1, from Jetson)")
        print("        • Firewall blocking port 8000 on the laptop")
        return False


# ── Step 2: WebSocket drone payload ──────────────────────────────────────────

async def check_websocket(server: str) -> bool:
    uri = f"ws://{server}:8000/ws/drone"   # ← correct endpoint
    print(f"\n[2/2] WebSocket test → {uri}")
    try:
        async with websockets.connect(uri, open_timeout=8) as ws:
            payload = build_payload()
            await ws.send(json.dumps(payload))
            print("      ✓  Connected & payload sent!")
            print(f"         yolo_detections : {payload['yolo_detections']}")
            print(f"         gdino_detections: {payload['gdino_detections']}")
            print(f"         gps             : {payload['gps']}")
            # Hold the connection briefly so the server logs the frame
            await asyncio.sleep(1.5)
            return True
    except OSError as exc:
        print(f"      ✗  Connection refused / unreachable: {exc}")
        print("         Check firewall rules and that the server is running.")
        return False
    except Exception as exc:
        print(f"      ✗  Unexpected error: {exc}")
        return False


# ── Main test runner ──────────────────────────────────────────────────────────

async def run_all(server: str) -> bool:
    print("=" * 60)
    print("  Hawk-I  Connection Test")
    print(f"  Target  : {server}:8000")
    print("=" * 60)

    # Step 1
    http_ok = check_http_health(server)
    if not http_ok:
        print("\n✗  Cannot reach backend — WebSocket test skipped.")
        return False

    # Step 2
    ws_ok = await check_websocket(server)

    # Summary
    print()
    print("=" * 60)
    if http_ok and ws_ok:
        print("  ✓  ALL CHECKS PASSED — Jetson → Laptop pipeline is working!")
    elif http_ok:
        print("  ✗  HTTP OK but WebSocket FAILED — check the server logs.")
    else:
        print("  ✗  FAILED — backend unreachable.")
    print("=" * 60)
    return http_ok and ws_ok


def main():
    parser = argparse.ArgumentParser(
        description="Hawk-I backend WebSocket connectivity test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  Laptop self-test:   python test_connection.py
  Jetson cross-test:  python test_connection.py --server 192.168.1.100
        """,
    )
    parser.add_argument(
        "--server",
        default="localhost",
        help="Backend server IP or hostname (default: localhost)",
    )
    args = parser.parse_args()

    ok = asyncio.run(run_all(args.server))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
