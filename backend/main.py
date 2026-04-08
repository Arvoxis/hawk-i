from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import json, asyncio, base64, logging, queue, threading, time, os
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO

from database import (
    init_db, save_detection,
    get_latest_detections, get_filtered_detections,
    update_detection_sam, update_detection_report, get_detection_report,
    CURRENT_TABLE, _SESSION_ID,
)
from pdf_generator import generate_inspection_pdf
from sam2_segmenter import SAM2Segmenter
from llm_reporter import LLMReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── GCS display / stats (integrated from gcs_client.py) ───────
_HEADLESS  = os.getenv("GCS_HEADLESS", "1") == "1"   # default: headless (no OpenCV window)
_SAVE_DIR  = os.getenv("GCS_SAVE_DIR",  "captures")
_AUTO_SAVE = os.getenv("GCS_AUTO_SAVE", "0") == "1"
_AUTO_SAVE_CONF = float(os.getenv("GCS_AUTO_SAVE_CONF", "0.70"))
_LOG_FILE  = os.getenv("GCS_LOG_FILE",  "detections.jsonl")

# Shared queues / events
_display_queue: queue.Queue = queue.Queue(maxsize=2)
_running = threading.Event()
_running.set()

# Live connection stats — updated by drone WebSocket handler
gcs_stats = {
    "connected":        False,
    "connect_time":     None,
    "drone_address":    None,
    "frames_received":  0,
    "total_detections": 0,
    "yolo_detections":  0,
    "gdino_detections": 0,
    "gs_detections":    0,
    "frames_saved":     0,
    "last_gps":         {"lat": None, "lon": None, "alt_m": None},
    "fps":              0.0,
}
_fps_times: list[float] = []


def _update_fps():
    now = time.time()
    _fps_times.append(now)
    while _fps_times and _fps_times[0] < now - 2.0:
        _fps_times.pop(0)
    gcs_stats["fps"] = len(_fps_times) / (now - _fps_times[0]) if len(_fps_times) >= 2 else 0.0


def _ensure_save_dir():
    os.makedirs(_SAVE_DIR, exist_ok=True)


def _save_frame(frame: np.ndarray, detections: list, gps: dict, reason: str = "auto"):
    _ensure_save_dir()
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_path = os.path.join(_SAVE_DIR, f"frame_{ts}.jpg")
    meta_path = os.path.join(_SAVE_DIR, f"frame_{ts}.json")
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = {"timestamp": ts, "reason": reason, "gps": gps, "detections": detections}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    gcs_stats["frames_saved"] += 1
    logger.info(f"Frame auto-saved: {img_path} ({len(detections)} detections)")


def _log_detections_jsonl(payload: dict):
    all_dets = payload.get("all_detections") or (
        payload.get("yolo_detections", []) + payload.get("gdino_detections", [])
    )
    if not all_dets:
        return
    entry = {
        "timestamp":  payload.get("timestamp", time.time()),
        "gps":        payload.get("gps", {}),
        "detections": all_dets,
    }
    try:
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Detection log write failed: {e}")


def _draw_stats_overlay(frame: np.ndarray, detections: list, gps: dict) -> np.ndarray:
    panel_h, panel_w = 220, 380
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 200, 255), 1)

    y = 35
    cv2.putText(frame, "HAWK-I GCS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    y += 30
    sc = (0, 255, 0) if gcs_stats["connected"] else (0, 0, 255)
    cv2.circle(frame, (25, y - 5), 5, sc, -1)
    cv2.putText(frame, "CONNECTED" if gcs_stats["connected"] else "WAITING",
                (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1)
    y += 25
    cv2.putText(frame, f"RX FPS : {gcs_stats['fps']:.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(frame, f"Frames : {gcs_stats['frames_received']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(frame, f"Detections: Y={gcs_stats['yolo_detections']} GD={gcs_stats['gdino_detections']} GS={gcs_stats['gs_detections']}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    y += 22
    lat = gps.get("lat"); lon = gps.get("lon"); alt = gps.get("alt_m")
    if lat is not None and lon is not None:
        gps_txt = f"GPS: {lat:.6f}, {lon:.6f}  Alt: {alt:.1f}m"
        gps_col = (0, 255, 200)
    else:
        gps_txt = "GPS: No fix"
        gps_col = (100, 100, 100)
    cv2.putText(frame, gps_txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, gps_col, 1)
    y += 22
    cv2.putText(frame, f"Saved: {gcs_stats['frames_saved']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    h = frame.shape[0]
    cv2.putText(frame, "[S] Save  [Q] Quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
    return frame


def _draw_waiting_screen() -> np.ndarray:
    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    cv2.putText(frame, "HAWK-I GCS",                    (230, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
    cv2.putText(frame, "Waiting for Jetson connection...", (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.putText(frame, "Backend: ws://0.0.0.0:8000/ws/drone", (185, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    dots = "." * (int(time.time() * 2) % 4)
    cv2.putText(frame, dots, (510, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    return frame


def _gcs_display_thread():
    """Background thread: shows live OpenCV window with drone feed + stats overlay."""
    if _HEADLESS:
        logger.info("GCS display: headless mode — no OpenCV window")
        return
    logger.info("GCS display thread started")
    cv2.namedWindow("Hawk-I GCS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hawk-I GCS", 1280, 720)
    last_fd = None
    while _running.is_set():
        try:
            last_fd = _display_queue.get(timeout=0.1)
        except queue.Empty:
            pass
        if last_fd is not None:
            frame = last_fd["frame"].copy()
            frame = _draw_stats_overlay(frame, last_fd["detections"], last_fd["gps"])
        else:
            frame = _draw_waiting_screen()
        cv2.imshow("Hawk-I GCS", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            _running.clear()
            break
        if key in (ord('s'), ord('S')) and last_fd is not None:
            _save_frame(last_fd["frame"], last_fd["detections"], last_fd["gps"], reason="manual")
    cv2.destroyAllWindows()
    logger.info("GCS display thread stopped")


def _gcs_stats_thread():
    """Periodic stats logger (useful when headless)."""
    while _running.is_set():
        time.sleep(3.0)
        if not _running.is_set():
            break
        gps = gcs_stats["last_gps"]
        gps_str = (
            f"{gps['lat']:.6f}, {gps['lon']:.6f} @ {(gps.get('alt_m') or 0):.1f}m"
            if gps.get("lat") is not None else "No fix"
        )
        logger.info(
            f"[GCS] Frames={gcs_stats['frames_received']} FPS={gcs_stats['fps']:.1f} "
            f"Dets(Y={gcs_stats['yolo_detections']} GD={gcs_stats['gdino_detections']} GS={gcs_stats['gs_detections']}) "
            f"Saved={gcs_stats['frames_saved']} GPS={gps_str} "
            f"{'CONNECTED' if gcs_stats['connected'] else 'WAITING'}"
        )

# ── Ground-station YOLO model (loaded at startup) ─────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_YOLO_WEIGHTS = _PROJECT_ROOT / "models" / "hawki_yolo11n.pt"
try:
    gs_yolo_model = YOLO(str(_YOLO_WEIGHTS))
    logger.info(f"Loaded ground-station YOLO weights from {_YOLO_WEIGHTS}")
except Exception as _yolo_err:
    gs_yolo_model = None
    logger.warning(f"Could not load GS YOLO weights ({_yolo_err}). GS inference disabled.")

# ── SAM2 segmenter (lazy-loads model on first request) ────────
segmenter = SAM2Segmenter()

# ── LLM reporter (calls Ollama / Gemma-3) ─────────────────────
reporter = LLMReporter()

# ── Startup: connect to DB + launch GCS background threads ───
@asynccontextmanager
async def lifespan(app):
    await init_db()
    for tgt, name in [(_gcs_display_thread, "GCS-Display"), (_gcs_stats_thread, "GCS-Stats")]:
        t = threading.Thread(target=tgt, name=name, daemon=True)
        t.start()
        logger.info(f"Started background thread: {name}")

    logger.info(f"Session started: {_SESSION_ID}  →  table: {CURRENT_TABLE}")

    yield

    # ── Auto-generate PDF report on shutdown ──────────────────
    _running.clear()
    logger.info("Generating session report on shutdown…")
    try:
        rows = await get_latest_detections(limit=10000)
        for r in rows:
            if r.get("detected_at"):
                r["detected_at"] = str(r["detected_at"])

        if rows:
            os.makedirs("reports", exist_ok=True)
            report_path = os.path.join("reports", f"hawki_{_SESSION_ID}.pdf")
            loop = asyncio.get_event_loop()
            pdf_bytes = await loop.run_in_executor(None, generate_inspection_pdf, rows)
            with open(report_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(f"Session report saved: {report_path}  ({len(rows)} detections)")
        else:
            logger.info("No detections this session — skipping report")
    except Exception as e:
        logger.error(f"Auto-report generation failed: {e}")

app = FastAPI(lifespan=lifespan)

dashboard_clients = []


# ── Helpers ───────────────────────────────────────────────────

def decode_jpeg(frame_b64: str) -> np.ndarray:
    """Decode a base64 JPEG string → RGB numpy array."""
    frame_bytes = base64.b64decode(frame_b64)
    frame_np    = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr   = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def run_gs_yolo(frame_rgb: np.ndarray) -> list[dict]:
    """Run ground-station YOLO inference on an RGB frame.

    Returns detections in the same format as the drone's yolo_detections
    so they can be merged into all_detections transparently.
    """
    if gs_yolo_model is None:
        return []
    results = gs_yolo_model(frame_rgb, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf            = float(box.conf[0])
        cls_id          = int(box.cls[0])
        class_name      = gs_yolo_model.names[cls_id]
        detections.append({
            "class":  class_name,
            "conf":   conf,
            "box":    [x1, y1, x2, y2],
            "source": "gs_yolo",
        })
    return detections


async def _run_sam2_and_update(frame_b64: str, saved: list[tuple[int, dict]], altitude_m: float):
    """Post-processing: decode frame → SAM2 → Gemma-3 report → update DB."""
    try:
        loop  = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, decode_jpeg, frame_b64)

        detections = [det for _, det in saved]
        results    = await loop.run_in_executor(
            None, segmenter.segment_detections, frame, detections, altitude_m
        )

        for (det_id, det), result in zip(saved, results):
            # ── 1. Save SAM2 area data ─────────────────────────────
            await update_detection_sam(
                det_id,
                area_px   = result["area_px"],
                area_cm2  = result["area_cm2"],
                sam_score = result["score"],
            )

            # ── 2. Build detection dict for the LLM ───────────────
            detection_data = {
                "detection_class": det.get("phrase") or det.get("class", "unknown"),
                "confidence":      det.get("conf", 0.0),
                "severity":        "L3" if det.get("conf", 0) > 0.85 else "L2" if det.get("conf", 0) > 0.65 else "L1",
                "area_cm2":        result["area_cm2"],
                "lat":             det.get("_lat", 0.0),
                "lon":             det.get("_lon", 0.0),
                "alt_m":           altitude_m,
                "sam_score":       result["score"],
            }

            # ── 3. Generate LLM report, save to DB ────────────────
            report = await reporter.generate_report(detection_data)
            await update_detection_report(det_id, report)

    except Exception as e:
        logger.error(f"SAM2/LLM post-processing failed: {e}")


# ── Drone WebSocket ───────────────────────────────────────────
@app.websocket("/ws/drone")
async def drone_receiver(websocket: WebSocket):
    """Receive detection payloads from the Jetson drone/client.

    Payload schema (all fields optional except gps / timestamp):
    {
        "timestamp":        float,
        "gps":              {"lat": float, "lon": float, "alt_m": float},
        "yolo_detections":  [{"class": str, "conf": float, "box": [x1,y1,x2,y2]}, ...],
        "gdino_detections": [{"phrase": str, "conf": float, "box": [...]}, ...],
        "frame_jpeg":       "<base64-encoded JPEG string>"   # optional
    }
    """
    await websocket.accept()
    client_addr = websocket.client.host if websocket.client else "unknown"
    logger.info(f"✓ Drone/Jetson connected from {client_addr}")

    # ── Update GCS live stats ─────────────────────────────────
    gcs_stats["connected"]     = True
    gcs_stats["connect_time"]  = time.time()
    gcs_stats["drone_address"] = client_addr

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            logger.info(f"RAW FRAME — keys={list(data.keys())} | gps={data.get('gps')} | yolo={len(data.get('yolo_detections', []))} | gdino={len(data.get('gdino_detections', []))} | has_frame={'yes' if data.get('frame_jpeg') else 'no'}")

            # ── Parse GPS (graceful defaults) ─────────────────
            gps = data.get("gps") or {}
            lat = gps.get("lat", 0.0)
            lon = gps.get("lon", 0.0)
            alt = gps.get("alt_m", 0.0)

            # ── Update GCS stats ──────────────────────────────
            yolo_dets  = data.get("yolo_detections",  [])
            gdino_dets = data.get("gdino_detections", [])
            gcs_stats["frames_received"]  += 1
            gcs_stats["yolo_detections"]  += len(yolo_dets)
            gcs_stats["gdino_detections"] += len(gdino_dets)
            gcs_stats["total_detections"] += len(yolo_dets) + len(gdino_dets)
            gcs_stats["last_gps"]          = gps
            _update_fps()

            # ── 0. Ground-station YOLO inference (backup / verification) ──
            frame_b64     = data.get("frame_jpeg")
            gs_detections: list[dict] = []
            decoded_frame: np.ndarray | None = None

            if frame_b64:
                loop          = asyncio.get_running_loop()
                decoded_frame = await loop.run_in_executor(None, decode_jpeg, frame_b64)
                gs_detections = await loop.run_in_executor(None, run_gs_yolo, decoded_frame)
                if gs_detections:
                    gcs_stats["gs_detections"] += len(gs_detections)
                    logger.info(f"GS-YOLO found {len(gs_detections)} detection(s)")

                # ── Push decoded frame to GCS display queue ────
                all_frame_dets = yolo_dets + gdino_dets + gs_detections
                display_data = {
                    "frame":      decoded_frame,
                    "detections": all_frame_dets,
                    "gps":        gps,
                    "timestamp":  data.get("timestamp", time.time()),
                }
                if _display_queue.full():
                    try:
                        _display_queue.get_nowait()
                    except queue.Empty:
                        pass
                _display_queue.put_nowait(display_data)

                # ── Forward frame + detections to dashboard WebSocket clients ──
                if dashboard_clients:
                    frame_msg = {
                        "frame_jpeg":  frame_b64,
                        "gps":         gps,
                        "detections":  all_frame_dets,
                        "timestamp":   data.get("timestamp", time.time()),
                    }
                    dead = []
                    for client in dashboard_clients:
                        try:
                            await client.send_json(frame_msg)
                        except Exception:
                            dead.append(client)
                    for c in dead:
                        dashboard_clients.remove(c)

                # ── Auto-save high-confidence frames ──────────
                if _AUTO_SAVE:
                    all_for_save = yolo_dets + gdino_dets + gs_detections
                    high = [d for d in all_for_save if d.get("conf", 0) >= _AUTO_SAVE_CONF]
                    if high:
                        loop.run_in_executor(
                            None, _save_frame, decoded_frame, all_for_save, gps, "auto"
                        )

            # ── Log detections to JSONL ────────────────────────
            _log_detections_jsonl({**data, "all_detections": yolo_dets + gdino_dets + gs_detections})

            all_detections = gdino_dets + yolo_dets + gs_detections

            # ── 1. Save every detection; collect (id, det) pairs ──────
            saved: list[tuple[int, dict]] = []

            for det in all_detections:
                if det.get("conf", 0) < 0.45:
                    continue

                class_name = det.get("phrase") or det.get("class", "unknown")
                confidence = float(det.get("conf", 0))
                severity   = "L3" if confidence > 0.85 else "L2" if confidence > 0.65 else "L1"

                det_id = await save_detection(
                    class_name=class_name,
                    confidence=confidence,
                    severity=severity,
                    area_cm2=0.0,
                    lat=lat, lon=lon,
                    altitude_m=alt,
                )
                det["_lat"] = lat
                det["_lon"] = lon
                saved.append((det_id, det))

                # Push initial pin to dashboard immediately
                pin = {
                    "lat":      lat,
                    "lon":      lon,
                    "class":    class_name,
                    "conf":     confidence,
                    "severity": severity,
                    "area_cm2": 0.0,
                }
                for client in dashboard_clients:
                    try:
                        await client.send_json(pin)
                    except Exception:
                        pass

            # ── 2. SAM2 post-processing (non-blocking) ────────────────
            if frame_b64 and saved:
                asyncio.create_task(
                    _run_sam2_and_update(frame_b64, saved, alt)
                )

    except WebSocketDisconnect:
        logger.info(f"✗ Drone disconnected from {client_addr}")
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON from drone: {e}")
    except Exception as e:
        logger.error(f"Drone WebSocket error: {e}")
    finally:
        gcs_stats["connected"]     = False
        gcs_stats["drone_address"] = None
        logger.info("Drone connection cleaned up")


# ── Dashboard WebSocket ───────────────────────────────────────
@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.append(websocket)
    logger.info(f"✓ Dashboard connected ({len(dashboard_clients)} total)")
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Dashboard WebSocket error: {e}")
    finally:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)
        logger.info(f"Dashboard disconnected ({len(dashboard_clients)} remaining)")


# ── REST endpoints ────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Hawk-I backend running ✓"}


@app.get("/health")
def health():
    """Lightweight health check — returns 200 if the server is up."""
    return {"ok": True, "timestamp": time.time()}


@app.get("/api/session")
def get_session():
    """Return the current session ID, active table, and live detection count."""
    return JSONResponse({
        "session_id":  _SESSION_ID,
        "table":       CURRENT_TABLE,
        "detections":  gcs_stats["total_detections"],
        "started_at":  gcs_stats["connect_time"],
    })


@app.get("/api/gcs/status")
def gcs_status():
    """Return live GCS connection stats (useful for testing connectivity)."""
    uptime_s = None
    if gcs_stats["connect_time"] and gcs_stats["connected"]:
        uptime_s = round(time.time() - gcs_stats["connect_time"], 1)
    return JSONResponse({
        **gcs_stats,
        "uptime_s": uptime_s,
    })


@app.get("/detections/latest")
async def get_latest(limit: int = 20):
    rows = await get_latest_detections(limit=limit)
    for r in rows:
        if r.get("detected_at"):
            r["detected_at"] = str(r["detected_at"])
    return JSONResponse(rows)


@app.get("/api/detections")
async def get_detections(
    limit: int = 100,
    class_name: str | None = None,
    severity: str | None = None,
):
    """Filtered detections. class_name and severity are comma-separated lists."""
    class_names = [c.strip() for c in class_name.split(",")] if class_name else None
    severities  = [s.strip() for s in severity.split(",")]   if severity  else None
    rows = await get_filtered_detections(limit=limit, class_names=class_names, severities=severities)
    for r in rows:
        if r.get("detected_at"):
            r["detected_at"] = str(r["detected_at"])
    return JSONResponse(rows)


@app.post("/query")
async def send_query(payload: dict):
    query_text = payload.get("query", "")
    print(f"  → Query received: {query_text}")
    return {"status": "query received", "query": query_text}


@app.get("/api/report/pdf")
async def download_pdf_report(
    severity:   str | None = None,
    class_name: str | None = None,
    limit:      int        = 500,
):
    """
    Generate and return a full inspection report PDF.
    Optional filters:
      ?severity=L3          — comma-separated, e.g. L3,L2
      ?class_name=crack     — comma-separated defect class names
    """
    severities  = [s.strip() for s in severity.split(",")]    if severity    else None
    class_names = [c.strip() for c in class_name.split(",")]  if class_name  else None

    rows = await get_filtered_detections(
        limit=limit,
        class_names=class_names,
        severities=severities,
    )
    for r in rows:
        if r.get("detected_at"):
            r["detected_at"] = str(r["detected_at"])

    loop     = asyncio.get_running_loop()
    pdf_bytes = await loop.run_in_executor(None, generate_inspection_pdf, rows)

    filename = "hawki_inspection_report.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/report/{detection_id}")
async def get_report(detection_id: int):
    """Return the LLM-generated inspection report for a given detection."""
    row = await get_detection_report(detection_id)
    if row is None:
        return JSONResponse({"error": "detection not found"}, status_code=404)
    if not row.get("llm_report"):
        return JSONResponse({"error": "report not yet generated"}, status_code=202)
    return JSONResponse({
        "id":         row["id"],
        "class_name": row["class_name"],
        "report":     row["llm_report"],
    })


class SegmentRequest(BaseModel):
    frame_jpeg:  str         # base64-encoded JPEG
    box:         list[float] # [x1, y1, x2, y2]
    altitude_m:  float


@app.post("/api/segment")
async def segment_single(req: SegmentRequest):
    """
    Accept a single base64 frame + bounding box + altitude.
    Return the SAM2-segmented area in cm².
    """
    try:
        loop   = asyncio.get_running_loop()
        frame  = await loop.run_in_executor(None, decode_jpeg, req.frame_jpeg)
        result = await loop.run_in_executor(
            None,
            segmenter.segment_box,
            frame,
            req.box,
        )
        area_cm2 = segmenter.px_to_cm2(
            result["area_px"], req.altitude_m, image_width_px=frame.shape[1]
        )
        return {
            "area_px":   result["area_px"],
            "area_cm2":  area_cm2,
            "sam_score": result["score"],
        }
    except Exception as e:
        logger.error(f"/api/segment failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
