from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json, asyncio, base64, logging
import numpy as np
import cv2

from database import init_db, save_detection, get_latest_detections, update_detection_sam
from sam2_segmenter import SAM2Segmenter

logger = logging.getLogger(__name__)

# ── SAM2 segmenter (lazy-loads model on first request) ────────
segmenter = SAM2Segmenter()

# ── Startup: connect to DB when server starts ─────────────────
@asynccontextmanager
async def lifespan(app):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)

dashboard_clients = []


# ── Helpers ───────────────────────────────────────────────────

def decode_jpeg(frame_b64: str) -> np.ndarray:
    """Decode a base64 JPEG string → RGB numpy array."""
    frame_bytes = base64.b64decode(frame_b64)
    frame_np    = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr   = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


async def _run_sam2_and_update(frame_b64: str, saved: list[tuple[int, dict]], altitude_m: float):
    """Post-processing: decode frame, run SAM2, write results back to DB."""
    try:
        loop  = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, decode_jpeg, frame_b64)

        detections = [det for _, det in saved]
        results    = await loop.run_in_executor(
            None, segmenter.segment_detections, frame, detections, altitude_m
        )

        for (det_id, _), result in zip(saved, results):
            await update_detection_sam(
                det_id,
                area_px   = result["area_px"],
                area_cm2  = result["area_cm2"],
                sam_score = result["score"],
            )

    except Exception as e:
        logger.error(f"SAM2 post-processing failed: {e}")


# ── Drone WebSocket ───────────────────────────────────────────
@app.websocket("/ws/drone")
async def drone_receiver(websocket: WebSocket):
    await websocket.accept()
    print("✓ Drone connected")
    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)

            all_detections = data["gdino_detections"] + data["yolo_detections"]
            alt            = data["gps"]["alt_m"]
            lat            = data["gps"]["lat"]
            lon            = data["gps"]["lon"]

            # ── 1. Save every detection; collect (id, det) pairs ──────
            saved: list[tuple[int, dict]] = []

            for det in all_detections:
                if det["conf"] < 0.45:
                    continue

                class_name = det.get("phrase") or det.get("class")
                confidence = det["conf"]
                severity   = "L3" if confidence > 0.85 else "L2" if confidence > 0.65 else "L1"

                det_id = await save_detection(
                    class_name=class_name,
                    confidence=confidence,
                    severity=severity,
                    area_cm2=0.0,       # SAM2 will update this below
                    lat=lat, lon=lon,
                    altitude_m=alt,
                )
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
            frame_b64 = data.get("frame_jpeg")
            if frame_b64 and saved:
                asyncio.create_task(
                    _run_sam2_and_update(frame_b64, saved, alt)
                )

    except WebSocketDisconnect:
        print("✗ Drone disconnected")


# ── Dashboard WebSocket ───────────────────────────────────────
@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.append(websocket)
    print(f"✓ Dashboard connected ({len(dashboard_clients)} total)")
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        dashboard_clients.remove(websocket)


# ── REST endpoints ────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Hawk-I backend running ✓"}


@app.get("/detections/latest")
async def get_latest():
    rows = await get_latest_detections(limit=20)
    for r in rows:
        if r.get("detected_at"):
            r["detected_at"] = str(r["detected_at"])
    return JSONResponse(rows)


@app.post("/query")
async def send_query(payload: dict):
    query_text = payload.get("query", "")
    print(f"  → Query received: {query_text}")
    return {"status": "query received", "query": query_text}


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
