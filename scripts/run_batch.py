#!/usr/bin/env python3
"""
run_batch.py — Run the full GCS pipeline over a folder of still images.

Until now the only way to put frames through Hawk-I was to have a drone (or a
simulator) streaming live. That makes the system a demo. This makes it a tool:
point it at any folder of photographs — archived inspection footage, a phone
walk-around, a downloaded corpus — and every image goes through the same
ingest path a real flight uses, producing the same database rows and the same
report.

It streams over the live /ws/drone WebSocket rather than importing the worker
directly, deliberately: that exercises the real ingest path, confidence gate,
processing queue and all, instead of a parallel code path that could drift.

    # Backend must be running with its own detector enabled:
    #   GS_YOLO_ENABLED=1 python run.py
    python scripts/run_batch.py --images data/real_frames
    python scripts/run_batch.py --images photos/ --altitude 8 --report out.pdf

Each image is sent with no edge detections attached, so the ground-station
YOLO pass is what finds the defects. Start the backend with GS_YOLO_ENABLED=1
or nothing will be detected.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import cv2                       # noqa: E402
import numpy as np               # noqa: E402
import websockets                # noqa: E402
import urllib.request            # noqa: E402

try:
    import config                # noqa: E402
    _DEFAULT_BASE = config.BACKEND_URL
except Exception:                # pragma: no cover
    _DEFAULT_BASE = "http://localhost:8000"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# A synthetic ground track. Batch stills have no GPS, but the database column
# is not nullable in practice and the map needs somewhere to put the pin, so
# frames are laid out along a short line near the reference site. Rows carry
# gps_synthetic in their source_model so this is never mistaken for a fix.
BASE_LAT, BASE_LON = 12.9716, 77.5946
STEP_DEG = 0.00012              # ~13 m between frames


def _find_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
    )


def _imread(path: Path):
    """Read an image, tolerating non-ASCII characters in the path.

    cv2.imread passes the filename to the C++ layer as a byte string, which on
    Windows silently fails for any path outside the active code page -- so
    "Château.jpg" and "Ausblühungen.jpg" were dropped without an error while
    the run still reported success. Reading the bytes in Python and decoding
    from memory sidesteps the filename entirely.
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _encode(path: Path, max_width: int) -> tuple[str, tuple[int, int]] | None:
    """Read an image and return (base64 JPEG, (w, h)), downscaled if needed.

    Width matters beyond bandwidth: the cm² conversion divides by frame width,
    so the width the backend decodes is the width the area is computed against.
    """
    img = _imread(path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode(), (w, h)


def _http_json(url: str, timeout: int = 15):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def stream(images: list[Path], args) -> int:
    ws_url = args.ws or f"ws://{args.host}:{args.port}/ws/drone"
    sent = 0

    print(f"\nStreaming {len(images)} image(s) → {ws_url}")
    async with websockets.connect(ws_url, open_timeout=15, max_size=32 * 1024 * 1024) as ws:
        for idx, path in enumerate(images):
            encoded = _encode(path, args.max_width)
            if encoded is None:
                print(f"  ! unreadable, skipped: {path.name}")
                continue
            b64, (w, h) = encoded

            payload = {
                "timestamp": time.time(),
                "gps": {
                    "lat":   BASE_LAT + idx * STEP_DEG,
                    "lon":   BASE_LON + idx * STEP_DEG * 0.6,
                    "alt_m": args.altitude,
                },
                # Empty: the ground-station detector is what should fire.
                "yolo_detections":  [],
                "gdino_detections": [],
                "frame_jpeg":       b64,
            }
            await ws.send(json.dumps(payload))
            sent += 1
            print(f"  [{sent}/{len(images)}] {path.name}  ({w}x{h})")
            await asyncio.sleep(args.interval)

    return sent


def summarise(base_url: str, before_ids: set, args) -> list[dict]:
    """Poll until processing settles, then print what was found."""
    print(f"\nWaiting up to {args.wait}s for the pipeline to drain…")

    deadline = time.time() + args.wait
    rows: list[dict] = []
    last_state: tuple[int, int] = (-1, -1)
    idle_polls = 0
    settle_polls = 5      # ~20s of no new rows before calling it done

    while time.time() < deadline:
        try:
            rows = _http_json(f"{base_url}/detections/latest?limit=200")
        except Exception as exc:
            print(f"  ! {exc}")
            time.sleep(3)
            continue

        new = [r for r in rows if r["id"] not in before_ids]
        done = [r for r in new if r.get("llm_report")]
        state = (len(new), len(done))

        # "All reported" alone is not a finish line: the first detection is
        # usually reported while later frames are still queued, so breaking on
        # it ended the run after one row.  The count also has to have stopped
        # moving.
        all_reported = bool(new) and len(done) == len(new)
        if all_reported and state == last_state and idle_polls >= settle_polls:
            break

        # Stop when nothing has changed for a while.  Waiting for every row to
        # carry a report deadlocked whenever a detection fell below
        # LLM_CONF_THRESHOLD and was never going to get one.
        # A CPU-bound LLM can take 30s+ per detection, so "no change" has to
        # mean minutes, not seconds, or the poll gives up mid-run.
        idle_polls = idle_polls + 1 if state == last_state else 0
        if idle_polls >= args.idle_polls:
            print(f"  no change for {idle_polls * 4}s, assuming the queue is drained"
                  + " " * 20)
            break
        last_state = state

        print(f"  {len(new)} detection(s), {len(done)} reported…".ljust(58), end="\r")
        time.sleep(4)

    new = [r for r in rows if r["id"] not in before_ids]
    print(" " * 60, end="\r")

    if not new:
        print("\nNo detections.")
        print("  The most common cause is the backend running without its own")
        print("  detector. Restart it with:  GS_YOLO_ENABLED=1 python run.py")
        return []

    print(f"\n{'id':<5} {'class':<24} {'conf':<6} {'sev':<4} {'area cm2':>11}  report")
    print("-" * 72)
    for r in sorted(new, key=lambda x: x["id"]):
        rep = r.get("llm_report") or ""
        tag = "—" if not rep else ("fallback" if "rule_based" in rep else "LLM")
        flag = "  ⚠ flagged" if r.get("dinov2_flagged") else ""
        print(f"{r['id']:<5} {r['class_name']:<24} {r['confidence']:.2f}   "
              f"{r['severity'] or '—':<4} {(r['area_cm2'] or 0):>11.1f}  {tag}{flag}")

    by_class: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for r in new:
        by_class[r["class_name"]] = by_class.get(r["class_name"], 0) + 1
        by_sev[r["severity"]] = by_sev.get(r["severity"], 0) + 1

    print(f"\n{len(new)} detection(s) across {len(by_class)} class(es)")
    print(f"  by class    : {by_class}")
    print(f"  by severity : {by_sev}")
    flagged = sum(1 for r in new if r.get("dinov2_flagged"))
    if flagged:
        print(f"  DINOv2 flagged {flagged} as probable false positive(s)")
    return new


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(_REPO_ROOT / "data" / "real_frames"),
                    help="Image file or directory (searched recursively)")
    ap.add_argument("--host", default=os.getenv("GCS_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("GCS_PORT", "8000")))
    ap.add_argument("--ws", default="", help="Override the full WebSocket URL")
    ap.add_argument("--altitude", type=float, default=10.0,
                    help="Altitude in metres to attribute to these frames; "
                         "drives the pixel→cm² conversion")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="Downscale wider images to this many pixels")
    ap.add_argument("--interval", type=float, default=1.5,
                    help="Seconds between frames; raise it if the queue backs up")
    ap.add_argument("--wait", type=int, default=1800,
                    help="Seconds to wait for processing to finish")
    ap.add_argument("--idle-polls", type=int, default=45,
                    help="Consecutive unchanged 4s polls before assuming the "
                         "queue is drained (~3 min at the default)")
    ap.add_argument("--report", default="",
                    help="Also download the PDF report to this path")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    root = Path(args.images)
    if not root.exists():
        print(f"No such path: {root}")
        print("Fetch a corpus first:  python scripts/fetch_real_frames.py")
        return 1

    images = _find_images(root)
    if not images:
        print(f"No images found under {root}")
        return 1

    try:
        health = _http_json(f"{base_url}/health")
    except Exception as exc:
        print(f"Backend unreachable at {base_url}: {exc}")
        print("Start it with:  GS_YOLO_ENABLED=1 python run.py")
        return 1

    print(f"Backend   : {base_url}  (session {health.get('session_id')})")
    print(f"Detector  : ground-station YOLO "
          f"{'ENABLED' if health.get('gs_yolo_enabled') else 'DISABLED'}")
    print(f"LLM       : {health.get('llm', {}).get('model')} "
          f"{'reachable' if health.get('llm', {}).get('reachable') else 'UNREACHABLE (fallback reports)'}")

    if not health.get("gs_yolo_enabled"):
        print("\n  Warning: the backend has no detector enabled, so these frames")
        print("  will produce no detections. Restart it with GS_YOLO_ENABLED=1.")

    try:
        before_ids = {r["id"] for r in _http_json(f"{base_url}/detections/latest?limit=500")}
    except Exception:
        before_ids = set()

    sent = asyncio.run(stream(images, args))
    print(f"\n{sent} frame(s) delivered.")

    new = summarise(base_url, before_ids, args)

    if args.report and new:
        try:
            print(f"\nDownloading report → {args.report}")
            with urllib.request.urlopen(f"{base_url}/api/report/pdf", timeout=300) as resp:
                Path(args.report).write_bytes(resp.read())
            size_kb = Path(args.report).stat().st_size / 1024
            print(f"  {size_kb:.1f} KB")
        except Exception as exc:
            print(f"  ! report download failed: {exc}")

    return 0 if new else 2


if __name__ == "__main__":
    sys.exit(main())
