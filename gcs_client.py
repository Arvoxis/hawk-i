"""
Hawk-I GCS Client — gcs_client.py
==================================
Runs on: GCS laptop (Windows/Linux/Mac)
Receives live video + detections from Jetson drone via WebSocket.

Features:
  - WebSocket server on /ws/drone (drone connects to this)
  - Live OpenCV window showing drone camera feed with detection boxes
  - Real-time stats overlay (FPS, detection counts, GPS, latency)
  - Keyboard commands for class updates and frame saving
  - Auto-saves high-confidence detection frames to disk
  - Logs all detections to JSON file for post-flight analysis

Keyboard Controls (in OpenCV window):
  S  — Save current frame to disk
  C  — Open class update prompt in terminal
  Q  — Quit

Usage:
  python gcs_client.py                                    # default: listen on 0.0.0.0:8765
  python gcs_client.py --port 8765 --save-dir captures    # custom port & save directory
  python gcs_client.py --headless                         # no OpenCV window (server only)
"""

import asyncio
import argparse
import base64
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import websockets

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("gcs")

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hawk-I GCS Client")
parser.add_argument("--host",       default="0.0.0.0",     help="Bind address")
parser.add_argument("--port",       default=8765, type=int, help="WebSocket port")
parser.add_argument("--save-dir",   default="captures",    help="Directory for saved frames")
parser.add_argument("--log-file",   default="detections.jsonl", help="Detection log file")
parser.add_argument("--headless",   action="store_true",   help="Run without OpenCV display")
parser.add_argument("--auto-save",  action="store_true",   help="Auto-save frames with detections")
parser.add_argument("--auto-save-conf", default=0.70, type=float,
                    help="Min confidence to trigger auto-save")
args = parser.parse_args()

# ── Shared state ──────────────────────────────────────────────────────────────

# Latest frame + metadata for display thread
_display_queue: queue.Queue = queue.Queue(maxsize=2)

# Command queue — display thread puts commands, server sends to drone
_command_queue: queue.Queue = queue.Queue(maxsize=10)

# Stats (thread-safe via GIL for simple reads)
_stats = {
    "frames_received":    0,
    "total_detections":   0,
    "yolo_detections":    0,
    "world_detections":   0,
    "frames_saved":       0,
    "last_gps":           {"lat": None, "lon": None, "alt_m": None},
    "last_timestamp":     0,
    "fps":                0.0,
    "connected":          False,
    "connect_time":       None,
}

# FPS tracker
_fps_times = []

# Global stop signal
_running = threading.Event()
_running.set()

# Connected drone WebSocket reference (for sending commands)
_drone_ws = None
_drone_ws_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_save_dir():
    os.makedirs(args.save_dir, exist_ok=True)


def save_frame(frame: np.ndarray, detections: list, gps: dict, reason: str = "manual"):
    """Save frame to disk with metadata sidecar JSON."""
    ensure_save_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_path  = os.path.join(args.save_dir, f"frame_{ts}.jpg")
    meta_path = os.path.join(args.save_dir, f"frame_{ts}.json")

    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    meta = {
        "timestamp":  ts,
        "reason":     reason,
        "gps":        gps,
        "detections": detections,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _stats["frames_saved"] += 1
    log.info(f"Frame saved: {img_path} ({reason}, {len(detections)} detections)")


def log_detections(payload: dict):
    """Append detections to JSONL log for post-flight analysis."""
    if not payload.get("all_detections"):
        return
    entry = {
        "timestamp":  payload["timestamp"],
        "gps":        payload["gps"],
        "detections": payload["all_detections"],
    }
    try:
        with open(args.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning(f"Failed to write detection log: {e}")


def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 JPEG string to OpenCV frame."""
    jpeg_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def update_fps():
    """Track receive FPS using a sliding window."""
    now = time.time()
    _fps_times.append(now)
    # Keep last 2 seconds of timestamps
    while _fps_times and _fps_times[0] < now - 2.0:
        _fps_times.pop(0)
    if len(_fps_times) >= 2:
        _stats["fps"] = len(_fps_times) / (now - _fps_times[0])
    else:
        _stats["fps"] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Server — Receives drone data
# ─────────────────────────────────────────────────────────────────────────────
async def handle_drone(ws, path):
    """Handle incoming drone WebSocket connection."""
    global _drone_ws

    if path != "/ws/drone":
        log.warning(f"Rejected connection on unknown path: {path}")
        await ws.close()
        return

    with _drone_ws_lock:
        _drone_ws = ws

    _stats["connected"] = True
    _stats["connect_time"] = time.time()
    log.info(f"Drone connected from {ws.remote_address}")

    try:
        # Start command sender alongside receiver
        await asyncio.gather(
            _receive_from_drone(ws),
            _send_commands_to_drone(ws),
        )
    except websockets.exceptions.ConnectionClosed as e:
        log.warning(f"Drone disconnected: {e}")
    except Exception as e:
        log.error(f"Drone handler error: {e}")
    finally:
        _stats["connected"] = False
        with _drone_ws_lock:
            _drone_ws = None
        log.info("Drone connection closed")


async def _receive_from_drone(ws):
    """Receive and process payloads from drone."""
    async for message in ws:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            log.warning(f"Invalid JSON from drone: {message[:100]}")
            continue

        # Update stats
        _stats["frames_received"] += 1
        yolo_count  = len(payload.get("yolo_detections", []))
        world_count = len(payload.get("world_detections", []))
        _stats["yolo_detections"]  += yolo_count
        _stats["world_detections"] += world_count
        _stats["total_detections"] += yolo_count + world_count
        _stats["last_gps"]         = payload.get("gps", _stats["last_gps"])
        _stats["last_timestamp"]   = payload.get("timestamp", 0)
        update_fps()

        # Decode frame
        frame_b64 = payload.get("frame_jpeg")
        if not frame_b64:
            continue

        frame = decode_frame(frame_b64)
        if frame is None:
            log.warning("Failed to decode frame JPEG")
            continue

        # Log detections to file
        log_detections(payload)

        # Auto-save if enabled and high-confidence detection found
        if args.auto_save:
            all_dets = payload.get("all_detections", [])
            high_conf = [d for d in all_dets if d["conf"] >= args.auto_save_conf]
            if high_conf:
                save_frame(frame, all_dets, payload.get("gps", {}), reason="auto")

        # Push to display queue (LIFO — drop stale)
        display_data = {
            "frame":      frame,
            "detections": payload.get("all_detections", []),
            "gps":        payload.get("gps", {}),
            "timestamp":  payload.get("timestamp", 0),
        }
        if _display_queue.full():
            try:
                _display_queue.get_nowait()
            except queue.Empty:
                pass
        _display_queue.put(display_data)


async def _send_commands_to_drone(ws):
    """Forward queued commands to drone."""
    while _running.is_set():
        try:
            cmd = _command_queue.get_nowait()
            await ws.send(json.dumps(cmd))
            log.info(f"Sent command to drone: {cmd}")
        except queue.Empty:
            await asyncio.sleep(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Display Thread — OpenCV live view
# ─────────────────────────────────────────────────────────────────────────────
def draw_stats_overlay(frame: np.ndarray, detections: list, gps: dict):
    """Draw transparent stats panel on top-left of frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark panel
    panel_h = 210
    panel_w = 360
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 200, 255), 1)

    # Title
    y = 35
    cv2.putText(frame, "HAWK-I GCS", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Connection status
    y += 30
    status_color = (0, 255, 0) if _stats["connected"] else (0, 0, 255)
    status_text  = "CONNECTED" if _stats["connected"] else "DISCONNECTED"
    cv2.circle(frame, (25, y - 5), 5, status_color, -1)
    cv2.putText(frame, status_text, (38, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # FPS
    y += 25
    cv2.putText(frame, f"RX FPS: {_stats['fps']:.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Frames received
    y += 22
    cv2.putText(frame, f"Frames: {_stats['frames_received']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Detection counts
    y += 22
    det_text = f"Detections: {len(detections)} (YOLO: {_stats['yolo_detections']} | World: {_stats['world_detections']})"
    cv2.putText(frame, det_text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # GPS
    y += 22
    lat = gps.get("lat")
    lon = gps.get("lon")
    alt = gps.get("alt_m")
    if lat is not None and lon is not None:
        gps_text = f"GPS: {lat:.6f}, {lon:.6f}  Alt: {alt:.1f}m"
        gps_color = (0, 255, 200)
    else:
        gps_text = "GPS: No fix"
        gps_color = (100, 100, 100)
    cv2.putText(frame, gps_text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, gps_color, 1)

    # Saved frames
    y += 22
    cv2.putText(frame, f"Saved: {_stats['frames_saved']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Keyboard hints at bottom
    hint_y = h - 15
    cv2.putText(frame, "[S] Save  [C] Classes  [Q] Quit", (10, hint_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


def draw_waiting_screen():
    """Show a waiting screen when no drone is connected."""
    frame = np.zeros((480, 720, 3), dtype=np.uint8)

    cv2.putText(frame, "HAWK-I GCS", (220, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)

    cv2.putText(frame, "Waiting for drone connection...", (180, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    port_text = f"Listening on {args.host}:{args.port}/ws/drone"
    cv2.putText(frame, port_text, (190, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    # Animated dots
    dots = "." * (int(time.time() * 2) % 4)
    cv2.putText(frame, dots, (490, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    cv2.putText(frame, "[Q] Quit", (320, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    return frame


def display_thread():
    """OpenCV display loop — shows live drone feed with stats overlay."""
    if args.headless:
        log.info("Headless mode — display disabled")
        return

    log.info("Display thread starting")
    cv2.namedWindow("Hawk-I GCS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hawk-I GCS", 1280, 720)

    last_frame_data = None

    while _running.is_set():
        # Try to get fresh frame
        try:
            last_frame_data = _display_queue.get(timeout=0.1)
        except queue.Empty:
            pass

        if last_frame_data is not None:
            frame = last_frame_data["frame"].copy()
            frame = draw_stats_overlay(
                frame,
                last_frame_data["detections"],
                last_frame_data["gps"]
            )
        else:
            frame = draw_waiting_screen()

        cv2.imshow("Hawk-I GCS", frame)

        key = cv2.waitKey(1) & 0xFF

        # Q — Quit
        if key == ord('q') or key == ord('Q'):
            log.info("Quit requested via keyboard")
            _running.clear()
            break

        # S — Save frame
        if (key == ord('s') or key == ord('S')) and last_frame_data is not None:
            save_frame(
                last_frame_data["frame"],
                last_frame_data["detections"],
                last_frame_data["gps"],
                reason="manual"
            )

        # C — Class update
        if key == ord('c') or key == ord('C'):
            # Non-blocking: launch input prompt in separate thread
            threading.Thread(
                target=_class_update_prompt,
                name="ClassInput",
                daemon=True
            ).start()

    cv2.destroyAllWindows()
    log.info("Display thread stopped")


def _class_update_prompt():
    """Terminal prompt for updating YOLO-World classes on the drone."""
    print("\n" + "=" * 50)
    print("  UPDATE YOLO-WORLD CLASSES")
    print("  Enter comma-separated classes (or 'cancel'):")
    print("=" * 50)
    try:
        user_input = input("> ").strip()
    except EOFError:
        return

    if not user_input or user_input.lower() == "cancel":
        print("Class update cancelled.")
        return

    new_classes = [c.strip() for c in user_input.split(",") if c.strip()]
    if not new_classes:
        print("No valid classes entered.")
        return

    cmd = {"classes": new_classes}
    _command_queue.put(cmd)
    print(f"Class update queued: {new_classes}")


# ─────────────────────────────────────────────────────────────────────────────
# Console Stats Thread — prints periodic stats (useful in headless mode)
# ─────────────────────────────────────────────────────────────────────────────
def stats_thread():
    """Print stats to console every 5 seconds."""
    while _running.is_set():
        time.sleep(5.0)
        if not _running.is_set():
            break

        gps = _stats["last_gps"]
        gps_str = (f"{gps['lat']:.6f}, {gps['lon']:.6f} @ {gps['alt_m']:.1f}m"
                   if gps.get("lat") is not None else "No fix")

        log.info(
            f"Stats — "
            f"RX: {_stats['frames_received']} frames | "
            f"FPS: {_stats['fps']:.1f} | "
            f"Dets: {_stats['total_detections']} "
            f"(YOLO: {_stats['yolo_detections']}, World: {_stats['world_detections']}) | "
            f"Saved: {_stats['frames_saved']} | "
            f"GPS: {gps_str} | "
            f"{'CONNECTED' if _stats['connected'] else 'WAITING'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
async def run_server():
    """Start WebSocket server and block until stopped."""
    log.info(f"Starting WebSocket server on {args.host}:{args.port}")

    server = await websockets.serve(
        handle_drone,
        args.host,
        args.port,
        ping_interval=15,
        ping_timeout=10,
        max_size=10 * 1024 * 1024,  # 10 MB — frames can be large
    )

    log.info(f"GCS listening on ws://{args.host}:{args.port}/ws/drone")

    # Block until _running is cleared
    while _running.is_set():
        await asyncio.sleep(0.5)

    server.close()
    await server.wait_closed()
    log.info("WebSocket server stopped")


def main():
    log.info("=" * 60)
    log.info("  Hawk-I Ground Control Station")
    log.info(f"  Listening  : {args.host}:{args.port}")
    log.info(f"  Save dir   : {args.save_dir}")
    log.info(f"  Log file   : {args.log_file}")
    log.info(f"  Headless   : {args.headless}")
    log.info(f"  Auto-save  : {args.auto_save} (conf >= {args.auto_save_conf})")
    log.info("=" * 60)

    ensure_save_dir()

    threads = [
        threading.Thread(target=display_thread, name="Display", daemon=True),
        threading.Thread(target=stats_thread,   name="Stats",   daemon=True),
    ]

    for t in threads:
        t.start()
        log.info(f"Started: {t.name}")

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        log.info("Shutting down...")
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        _running.clear()

    for t in threads:
        t.join(timeout=3.0)

    log.info("GCS shut down cleanly.")


if __name__ == "__main__":
    main()
