"""
sam3_worker.py — SAM 2 bounding-box segmentation, severity-coloured mask overlay,
and annotated-frame persistence.

GPU optimisation: process_frame() encodes the image ONCE on the GPU and
predicts all bounding boxes in a single inference context — N× faster than
calling segment_box() once per detection (which re-runs the heavy image
encoder each time).
"""

import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Camera / GSD constants ────────────────────────────────────────────────────
_SENSOR_W_MM   = 6.287
_FOCAL_MM      = 4.74
_IMG_W_PX      = 1920
_DEFAULT_ALT_M = 10.0
_MIN_ALT_M     = 2.0    # altitudes below this are treated as bad GPS

# ── Severity thresholds (area-based) ─────────────────────────────────────────
_L3_CM2 = 500.0
_L2_CM2 = 100.0

# ── Mask quality gate — below this IoU score, fall back to bbox area ─────────
_MASK_QUALITY_THRESHOLD = 0.75

# ── Overlay colours (BGR tuples) ──────────────────────────────────────────────
_SEV_BGR = {
    "L1": (0,   200, 0),    # green
    "L2": (0,   165, 255),  # orange
    "L3": (50,  50,  255),  # red
}

# ── Lazy SAM2 segmenter singleton ─────────────────────────────────────────────
_segmenter = None


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        from sam2_segmenter import SAM2Segmenter
        _segmenter = SAM2Segmenter()
    return _segmenter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_altitude(gps: dict) -> float:
    """Return a safe altitude value, warning if the raw value is bad."""
    raw = (gps or {}).get("alt_m")
    try:
        alt = float(raw)
    except (TypeError, ValueError):
        alt = 0.0

    if raw is None or alt < _MIN_ALT_M:
        logger.warning(
            "alt_m=%s is null or <%.1f m — GSD calculation unreliable, "
            "using default %.1f m",
            raw, _MIN_ALT_M, _DEFAULT_ALT_M,
        )
        return _DEFAULT_ALT_M
    return alt


def _gsd_cm_per_px(alt_m: float, img_w_px: int = _IMG_W_PX) -> float:
    """Ground Sampling Distance in cm/px. Pass the actual decoded frame width."""
    return (alt_m * _SENSOR_W_MM) / (_FOCAL_MM * img_w_px) * 10


def _classify(area_cm2: float) -> str:
    if area_cm2 >= _L3_CM2:
        return "L3"
    if area_cm2 >= _L2_CM2:
        return "L2"
    return "L1"


def _draw_detection(
    canvas_bgr: np.ndarray,
    mask,
    box: list,
    class_name: str,
    severity: str,
    conf: float = 0.0,
    area_cm2: float = 0.0,
) -> None:
    """Draw mask overlay + bounding box + label on a BGR canvas in-place.

    Blend strategy:
        overlay = zeros the same shape as canvas_bgr
        overlay[mask] = severity colour
        canvas_bgr = addWeighted(canvas_bgr, 0.6, overlay, 0.4, 0)

    Non-mask pixels: canvas unchanged (0.6 * canvas + 0.4 * 0 = 0.6 * canvas).
    Mask pixels: 0.6 * original + 0.4 * colour → clearly tinted, not washed out.
    """
    bgr_col = _SEV_BGR[severity]

    if mask is not None and mask.any():
        overlay = np.zeros_like(canvas_bgr)
        overlay[mask] = bgr_col
        # safe in-place: addWeighted reads src1 before writing dst
        cv2.addWeighted(canvas_bgr, 0.6, overlay, 0.4, 0, canvas_bgr)
    else:
        logger.warning(
            "_draw_detection: mask is empty for %s — no colour overlay drawn", class_name
        )

    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), bgr_col, 2)

    label = f"{class_name} {severity} {conf:.0%} {area_cm2:.0f}cm2"
    label_y = max(y1 - 8, 16)
    # dark background strip behind text for legibility
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(canvas_bgr,
                  (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2),
                  (0, 0, 0), -1)
    cv2.putText(canvas_bgr, label,
                (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, bgr_col, 1,
                cv2.LINE_AA)


# ── Primary API: process entire frame in ONE GPU encode call ──────────────────

def process_frame(
    frame_np:   np.ndarray,
    detections: list[dict],
    gps:        dict,
) -> tuple[list[dict], bytes | None]:
    """
    Run SAM 2 on ALL bounding boxes in a single frame using one image-encode
    call.  This is the GPU-efficient path — the heavy ViT encoder runs once
    regardless of how many detections are in the frame.

    Parameters
    ----------
    frame_np   : RGB numpy array (H × W × 3).
    detections : list of {"box": [x1,y1,x2,y2], "class_name": str, "conf": float}
    gps        : {"lat": float, "lon": float, "alt_m": float}

    Returns
    -------
    (results, composite_jpeg_bytes)
        results               : list of per-detection dicts.
        composite_jpeg_bytes  : JPEG bytes of the frame with ALL masks drawn.
                                None on error.
    """
    if not detections:
        logger.debug("process_frame: no detections — skipping SAM2")
        return [], None

    # ── Frame validation ──────────────────────────────────────────────────────
    if frame_np is None or frame_np.size == 0:
        logger.error("process_frame: frame_np is None or empty — aborting")
        return [], None

    h, w = frame_np.shape[:2]
    logger.debug("process_frame: input frame shape=%s dtype=%s", frame_np.shape, frame_np.dtype)

    # ── Altitude / GSD — use actual frame width, not the hardcoded constant ──
    alt_m = _resolve_altitude(gps)
    gsd   = _gsd_cm_per_px(alt_m, img_w_px=w)
    logger.debug("process_frame: alt_m=%.1f gsd=%.4f cm/px (frame %dx%d)", alt_m, gsd, w, h)

    seg = _get_segmenter()
    seg._load_model()

    bgr_base  = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    composite = bgr_base.copy()

    os.makedirs(os.path.join("data", "frames"), exist_ok=True)

    results: list[dict] = []

    # ── Single GPU image encode, N box predictions ────────────────────────────
    with torch.inference_mode(), seg._autocast_ctx():
        seg._predictor.set_image(frame_np)
        logger.debug("process_frame: SAM2 image encoder done (%d detection(s))", len(detections))

        for det in detections:
            box        = det.get("box", [0, 0, 64, 64])
            class_name = det.get("class_name", "defect")
            conf       = float(det.get("conf", 0.0))

            # ── Validate box ──────────────────────────────────────────────
            if len(box) != 4:
                logger.error("SAM2 ▶ %s: invalid box %s — skipping", class_name, box)
                continue

            x1, y1, x2, y2 = [float(v) for v in box]

            # Detect normalised coordinates (0–1 range) and scale to pixels
            if max(x1, y1, x2, y2) <= 1.0:
                logger.warning(
                    "SAM2 ▶ %s: box %s looks normalised (max=%.3f) — "
                    "scaling to pixel space (%dx%d)",
                    class_name, box, max(x1, y1, x2, y2), w, h,
                )
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
                box = [x1, y1, x2, y2]

            box_area = (x2 - x1) * (y2 - y1)
            if box_area <= 0:
                logger.error(
                    "SAM2 ▶ %s: degenerate box %s (area=%.1f px²) — skipping",
                    class_name, box, box_area,
                )
                continue

            logger.debug(
                "SAM2 ▶ %s: box=[%.1f,%.1f,%.1f,%.1f] area=%.0f px²",
                class_name, x1, y1, x2, y2, box_area,
            )

            # ── SAM2 predict ──────────────────────────────────────────────
            n_masks = 0
            try:
                box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                masks, scores, _ = seg._predictor.predict(
                    box=box_np,
                    multimask_output=False,
                )

                n_masks = len(masks) if masks is not None else 0
                if n_masks == 0:
                    logger.error(
                        "SAM2 ▶ %s: predict() returned 0 masks — "
                        "check box coords and model weights", class_name,
                    )
                    mask    = np.zeros((h, w), dtype=bool)
                    score   = 0.0
                    area_px = 0
                else:
                    mask    = masks[0].astype(bool)
                    score   = float(scores[0])
                    area_px = int(mask.sum())

                    logger.debug(
                        "SAM2 ▶ %s: n_masks=%d best_score=%.3f area_px=%d",
                        class_name, n_masks, score, area_px,
                    )

                    if area_px == 0:
                        # Mask is all-False: box was out-of-frame or SAM confused.
                        # Use the bounding box pixel area as a conservative fallback.
                        bbox_area_px = int((x2 - x1) * (y2 - y1))
                        logger.warning(
                            "SAM2 ▶ %s: mask returned but area_px=0 — "
                            "falling back to bbox area %d px²",
                            class_name, bbox_area_px,
                        )
                        area_px = bbox_area_px
                        score   = -1.0   # mark as bbox-estimate, not real mask
                    if score >= 0 and score < 0.5:
                        logger.warning(
                            "SAM2 ▶ %s: low IoU score %.3f — mask quality poor",
                            class_name, score,
                        )

                    # ── Mask quality gate ─────────────────────────────────
                    # When SAM2 confidence is below threshold, use bbox pixel
                    # count for area calculation (conservative estimate).
                    if score < _MASK_QUALITY_THRESHOLD and area_px > 0:
                        bbox_area_px = int((x2 - x1) * (y2 - y1))
                        logger.warning(
                            "SAM2 ▶ %s: score %.3f < %.2f — "
                            "using bbox area %d px² instead of mask %d px²",
                            class_name, score, _MASK_QUALITY_THRESHOLD,
                            bbox_area_px, area_px,
                        )
                        area_px = bbox_area_px

            except Exception as exc:
                logger.warning(
                    "SAM2 ▶ %s predict failed for box=%s: %s — bbox fill fallback",
                    class_name, box, exc,
                )
                mask    = np.zeros((h, w), dtype=bool)
                mask[max(0, int(y1)):max(0, int(y2)),
                     max(0, int(x1)):max(0, int(x2))] = True
                area_px = int(mask.sum())
                score   = 0.0

            # ── Area + severity ───────────────────────────────────────────
            area_cm2 = float(area_px) * (gsd ** 2)
            severity = _classify(area_cm2)

            logger.info(
                "SAM2 ▶ %s | box=[%.0f,%.0f,%.0f,%.0f] | "
                "n_masks=%d score=%.3f area_px=%d | "
                "alt_m=%.1f gsd=%.4f area_cm2=%.1f | sev=%s",
                class_name, x1, y1, x2, y2,
                n_masks, score, area_px,
                alt_m, gsd, area_cm2, severity,
            )

            # ── Individual annotated image — only save L2/L3 to disk ─────
            individual = bgr_base.copy()
            _draw_detection(individual, mask, box, class_name, severity, conf, area_cm2)

            img_path = None
            if severity in ("L2", "L3"):
                ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                safe_cls = class_name.replace(" ", "_")
                img_path = os.path.join("data", "frames", f"{ts}_{safe_cls}.jpg")
                ok = cv2.imwrite(img_path, individual, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    logger.error("SAM2 ▶ %s: cv2.imwrite failed for path %s", class_name, img_path)
                    img_path = None
                else:
                    logger.debug("SAM2 ▶ %s: annotated frame saved → %s", class_name, img_path)

            # ── Draw on composite frame for MJPEG stream ──────────────────
            _draw_detection(composite, mask, box, class_name, severity, conf, area_cm2)

            results.append({
                "mask_image_path": img_path,
                "area_cm2":        round(area_cm2, 2),
                "severity":        severity,
                "pixel_count":     area_px,
                "sam_score":       score,
                # Raw mask returned so DINOv2 can isolate the defect crop.
                # Stored as bool numpy array; caller decides whether to keep it.
                "mask":            mask,
            })

    # ── Encode composite frame for MJPEG stream ───────────────────────────────
    if results:
        ok, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            logger.error("process_frame: cv2.imencode failed for composite frame")
            composite_jpeg = None
        else:
            composite_jpeg = buf.tobytes()
            logger.debug(
                "process_frame: composite JPEG encoded (%d bytes, %d detection(s))",
                len(composite_jpeg), len(results),
            )
    else:
        composite_jpeg = None

    return results, composite_jpeg


# ── Legacy single-detection path (kept for /api/segment endpoint) ─────────────

def process_detection(
    frame_np:   np.ndarray,
    box:        list,
    gps:        dict,
    class_name: str = "defect",
) -> dict:
    """Single-detection wrapper around process_frame (for backward compat)."""
    results, jpeg = process_frame(
        frame_np,
        [{"box": box, "class_name": class_name, "conf": 0.0}],
        gps,
    )
    if results:
        r = results[0]
        return {**r, "annotated_jpeg_bytes": jpeg}
    return {
        "mask_image_path":      "",
        "annotated_jpeg_bytes": None,
        "area_cm2":             0.0,
        "severity":             "L1",
        "pixel_count":          0,
        "sam_score":            0.0,
    }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Find test frame ───────────────────────────────────────────────────────
    test_dir = Path("data/test_frames")
    jpegs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg"))
    if not jpegs:
        print(f"ERROR: no JPEG files found in {test_dir.resolve()}")
        print("Place at least one JPEG in data/test_frames/ and re-run.")
        sys.exit(1)

    src = jpegs[0]
    print(f"\n[test] Loading frame: {src}")
    bgr = cv2.imread(str(src))
    if bgr is None:
        print(f"ERROR: cv2.imread failed for {src}")
        sys.exit(1)

    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    print(f"[test] Frame shape: {frame_rgb.shape}")

    # ── Bounding box = centre third of the image ──────────────────────────────
    x1 = w // 3
    y1 = h // 3
    x2 = 2 * w // 3
    y2 = 2 * h // 3
    box = [x1, y1, x2, y2]
    print(f"[test] Using bounding box (centre third): {box}")

    # ── Run SAM2 via process_frame ────────────────────────────────────────────
    gps = {"lat": 12.9716, "lon": 77.5946, "alt_m": 12.5}
    results, jpeg_bytes = process_frame(
        frame_rgb,
        [{"box": box, "class_name": "test_defect", "conf": 0.90}],
        gps,
    )

    # ── Save output ───────────────────────────────────────────────────────────
    out_dir = Path("data/frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sam2_test_output.jpg"

    if jpeg_bytes:
        out_path.write_bytes(jpeg_bytes)
        print(f"\n[test] ✓ Composite MJPEG frame saved → {out_path.resolve()}")
    else:
        print("\n[test] WARNING: no JPEG bytes returned — check logs above")

    if results:
        r = results[0]
        print(f"[test] area_cm2  = {r['area_cm2']} cm²")
        print(f"[test] severity  = {r['severity']}")
        print(f"[test] pixel_count = {r['pixel_count']} px")
        print(f"[test] sam_score = {r['sam_score']:.3f}")
        print(f"[test] individual frame → {r['mask_image_path']}")
    else:
        print("[test] ERROR: process_frame returned no results")
