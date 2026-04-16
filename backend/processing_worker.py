"""
processing_worker.py — Async queue-based pipeline for SAM2 + LLM processing.

Each item in raw_queue is a frame-level dict pushed by the /ws/drone receiver:
    {
        "timestamp":        float,
        "gps":              {"lat": float, "lon": float, "alt_m": float},
        "yolo_detections":  [...],   # already filtered ≥ 0.45 conf
        "gdino_detections": [...],   # already filtered ≥ 0.45 conf
        "frame_np":         np.ndarray | None,  # RGB array, decoded in receiver
    }

Pipeline per frame:
    1. Persist all detections as raw DB rows (no frame bytes — metadata only)
    2. Run SAM2 once for the whole frame in a thread-pool thread (GPU batch)
    3. Update each row with area_cm2 / severity / image_path / sam_score
    4. Generate an LLM report for high-confidence detections
"""

import asyncio
import json
import logging
import math

import numpy as np

import database as _db
import sam3_worker
from llm_worker import _call_ollama, _fallback, CONF_THRESH
from database import save_detection_raw
from dinov2_embedder import get_embedder, LOW_SIMILARITY_THRESHOLD


def _estimate_area_cm2_from_box(box: list, alt_m: float) -> float:
    """Estimate defect area in cm² from a bounding box.

    Uses the same GSD formula as sam3_worker.
    When altitude is missing or unreliable (< 2 m), uses 10 m as a safe default
    so the estimate is always non-zero for a valid box.
    """
    if not box or len(box) != 4:
        return 0.0
    # Use a safe default altitude so we never return 0 for a valid box
    effective_alt = alt_m if alt_m >= 2.0 else 10.0
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
        box_w_px = abs(x2 - x1)
        box_h_px = abs(y2 - y1)
        area_px  = box_w_px * box_h_px
        if area_px <= 0:
            return 0.0
        # Match sam3_worker sensor constants
        sensor_w_mm = 6.287
        focal_mm    = 4.74
        img_w_px    = 1280
        gsd = (effective_alt * sensor_w_mm) / (focal_mm * img_w_px) * 10  # cm/px
        return round(area_px * (gsd ** 2), 2)
    except Exception:
        return 0.0

logger = logging.getLogger(__name__)

# ── Shared queue ───────────────────────────────────────────────────────────────
raw_queue: asyncio.Queue = asyncio.Queue()
_processed_total: int = 0


async def _process_one(raw_item: dict) -> None:
    """Full pipeline for one frame-level dict from raw_queue."""
    loop = asyncio.get_running_loop()

    gps = raw_item.get("gps", {})
    lat = float(gps.get("lat") or 0.0)
    lon = float(gps.get("lon") or 0.0)
    alt = float(gps.get("alt_m") or 0.0)

    yolo_dets  = raw_item.get("yolo_detections", [])
    gdino_dets = raw_item.get("gdino_detections", [])
    all_dets   = yolo_dets + gdino_dets
    frame_np   = raw_item.get("frame_np")

    if not all_dets:
        return

    # ── 1. Persist all detections as raw rows (no frame bytes) ────────────────
    # saved: list of (det_id, class_name, confidence, box, source)
    saved = []
    for det in all_dets:
        class_name = det.get("phrase") or det.get("class", "unknown")
        confidence = float(det.get("conf", 0))
        box        = det.get("box", [])
        src        = det.get("source", "yolo_world")
        if "phrase" in det and src == "yolo_world":
            src = "grounding_dino"

        det_id = await save_detection_raw(
            class_name   = class_name,
            confidence   = confidence,
            lat          = lat,
            lon          = lon,
            altitude_m   = alt,
            source_model = src,
            raw_box_json = json.dumps(box),
        )
        saved.append((det_id, class_name, confidence, box))

    # ── 2. Run SAM2 once for the whole frame (one GPU encode) ─────────────────
    sam_results = []
    if frame_np is not None and frame_np.size > 0:
        dets_for_sam = [
            {"box": box, "class_name": cls, "conf": conf}
            for _, cls, conf, box in saved
        ]
        try:
            sam_results, _ = await loop.run_in_executor(
                None,
                sam3_worker.process_frame,
                frame_np,
                dets_for_sam,
                {"lat": lat, "lon": lon, "alt_m": alt},
            )
        except Exception as exc:
            logger.warning("SAM2 batch failed: %s — using confidence-based severity", exc)

    # ── 3. Update each row with SAM2 results (or fallback severity) ────────────
    embedder = get_embedder()
    _sev_downgrade = {"L3": "L2", "L2": "L1", "L1": "L1"}

    for i, (det_id, class_name, confidence, box) in enumerate(saved):
        if i < len(sam_results):
            r        = sam_results[i]
            severity = r["severity"]
            area_cm2 = r["area_cm2"]
            sam_mask = r.get("mask")     # bool numpy array or None
            await _db.update_detection_sam(
                det_id,
                area_px    = r.get("pixel_count", 0),
                area_cm2   = area_cm2,
                sam_score  = r["sam_score"],
                image_path = r.get("mask_image_path"),
                severity   = severity,
            )
        else:
            severity = "L3" if confidence > 0.85 else "L2" if confidence > 0.65 else "L1"
            # Estimate area from bounding box when no frame was provided.
            # sam_score=-1 signals to the PDF generator that this is an estimate.
            area_cm2 = _estimate_area_cm2_from_box(box, alt)
            sam_mask = None
            await _db.update_detection_sam(
                det_id, 0, area_cm2,
                sam_score=-1.0,   # sentinel: bbox-estimated, not SAM-measured
                image_path=None,
                severity=severity,
            )

        # ── 3b. SAM false-positive pre-check ──────────────────────────────────
        # A very low SAM confidence on a tiny area almost always means the
        # detector fired on texture/shadow rather than a real defect.
        sam_fp = False
        if i < len(sam_results):
            s_score = sam_results[i].get("sam_score", 1.0)
            s_area  = sam_results[i].get("area_cm2", 999.0)
            if s_score < 0.25 and s_area < 15.0:
                sam_fp = True
                logger.info(
                    "SAM FP ▶ %s det_id=%d: sam_score=%.3f area=%.1fcm² → probable false positive",
                    class_name, det_id, s_score, s_area,
                )

        # ── 3c. DINOv2 embedding + confidence re-scoring ──────────────────────
        similar_list: list[dict] = []
        _dinov2_flagged_local: bool = sam_fp          # inherit SAM FP flag
        if frame_np is not None and frame_np.size > 0:
            try:
                # embed_crop is CPU/GPU-intensive → run in thread pool
                embedding: np.ndarray = await loop.run_in_executor(
                    None, embedder.embed_crop, frame_np, sam_mask, box
                )

                # ── Centroid check (reliable once ≥5 examples exist) ──────────
                centroid = await embedder.get_class_centroid(class_name)
                dinov2_flagged = sam_fp   # carry forward SAM flag
                if centroid is not None:
                    sim = embedder.cosine_similarity(embedding, centroid)
                    if sim < LOW_SIMILARITY_THRESHOLD:        # 0.45
                        dinov2_flagged = True
                        _dinov2_flagged_local = True
                        severity = _sev_downgrade.get(severity, severity)
                        logger.info(
                            "DINOv2 centroid ▶ %s det_id=%d: sim=%.3f < %.2f "
                            "→ flagged, severity → %s",
                            class_name, det_id, sim, LOW_SIMILARITY_THRESHOLD, severity,
                        )

                # ── Peer check: works from the FIRST detection onward ─────────
                # If the closest past example of this class is still very
                # dissimilar, the detection is probably a false positive even
                # before we have enough data for a centroid.
                similar_list = await embedder.find_similar(embedding, class_name, top_k=3)
                if similar_list:
                    best_sim = max(s["similarity_score"] for s in similar_list)
                    # Hard FP threshold — stricter than centroid soft-flag
                    if best_sim < 0.20:
                        dinov2_flagged = True
                        _dinov2_flagged_local = True
                        severity = "L1"   # lowest severity; operator must verify
                        logger.info(
                            "DINOv2 peer ▶ %s det_id=%d: best_sim=%.3f < 0.20 "
                            "→ probable false positive, severity → L1",
                            class_name, det_id, best_sim,
                        )

                similar_json = json.dumps([s["id"] for s in similar_list]) if similar_list else None

                # Invalidate centroid cache now that a new embedding exists
                embedder.invalidate_centroid(class_name)

                await _db.update_detection_dinov2(
                    det_id,
                    embedding_bytes     = embedding.tobytes(),
                    confidence_adjusted = confidence,
                    dinov2_flagged      = dinov2_flagged,
                    severity            = severity if dinov2_flagged else None,
                    similar_ids         = similar_json,
                )
            except Exception as exc:
                logger.warning(
                    "DINOv2 pipeline failed for det_id=%d: %s — skipping", det_id, exc
                )

        # ── 4. LLM report for high-confidence detections ──────────────────────
        if confidence >= CONF_THRESH:
            similar_note = ""
            if similar_list:
                similar_note = "\nSimilar past detections:\n" + "\n".join(
                    f"  - id={s['id']} at ({s['lat']:.5f},{s['lon']:.5f}) "
                    f"on {s['detected_at'][:10]} sim={s['similarity_score']:.2f}"
                    for s in similar_list
                )
            dino_note = (
                "DINOv2+SAM checks FAILED — probable false positive, verify manually"
                if (sam_fp and _dinov2_flagged_local)
                else "DINOv2 similarity check FAILED — verify manually"
                if _dinov2_flagged_local
                else "SAM score low — verify manually"
                if sam_fp
                else "DINOv2+SAM checks passed"
            )
            # Skip LLM report entirely for high-confidence false positives
            # (both SAM and DINOv2 agree it's wrong) to save inference cost.
            if sam_fp and _dinov2_flagged_local:
                logger.info(
                    "FP consensus ▶ det_id=%d: skipping LLM report (SAM+DINOv2 both failed)",
                    det_id,
                )
                continue
            prompt = (
                f"Defect class: {class_name}\n"
                f"Detections in the last 30 s: 1\n"
                f"Average confidence: {confidence:.2f}\n"
                f"Average area: {area_cm2:.1f} cm²\n"
                f"Observed severities: {severity}\n"
                f"Location: Bengaluru, Karnataka, India\n"
                f"DINOv2 note: {dino_note}"
                f"{similar_note}\n"
                "Generate the inspection report JSON."
            )
            try:
                report = await _call_ollama(prompt)
            except Exception as exc:
                logger.warning("LLM call failed for det %d: %s — fallback", det_id, exc)
                report = _fallback(class_name, confidence, 1)

            await _db.update_detection_report(det_id, json.dumps(report))

    logger.info(
        "Worker ▶ frame: %d detection(s) | SAM2=%s | GPS=(%.4f,%.4f)",
        len(saved), "yes" if sam_results else "no", lat, lon,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_processing_worker() -> None:
    """Drain raw_queue continuously.  Runs as an asyncio background task."""
    global _processed_total
    logger.info(
        "Processing worker started (LLM threshold=%.2f, queue=asyncio.Queue)",
        CONF_THRESH,
    )

    while True:
        raw_item = await raw_queue.get()

        try:
            await _process_one(raw_item)
            _processed_total += 1

            if _processed_total % 10 == 0:
                logger.info(
                    "Processing queue depth: %d remaining  (total processed: %d)",
                    raw_queue.qsize(), _processed_total,
                )
        except Exception as exc:
            logger.error("Unhandled error processing frame: %s", exc)
        finally:
            raw_queue.task_done()

        await asyncio.sleep(0.1)
