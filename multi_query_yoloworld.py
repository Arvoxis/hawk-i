"""
multi_query_yoloworld.py — Multi-query YOLO-World wrapper for defect detection.

YOLO-World uses CLIP text embeddings internally.  Plain-language descriptions
of each defect class ("thin line in concrete") yield significantly better recall
than single technical labels ("Crack"), because CLIP was trained on natural
image-text pairs, not domain jargon.

Architecture
------------
All sub-queries from all classes are flattened into a *single* set_classes() call.
YOLO-World's set_classes() encodes every string in one CLIP text-encoder forward
pass at init time.  Each predict() call then runs one vision forward pass regardless
of how many classes or sub-queries are defined.

Total cost per frame:
    1× CLIP text encode  (amortised at __init__)
  + 1× vision forward   (every predict() call)

This is NOT one forward pass per sub-query — batching is free.

YOLO-World batching limitation
-------------------------------
set_classes() accepts at most ~80 classes before CLIP context overflow degrades
results.  The default QUERY_MAP uses 30 sub-queries total (6 classes × 5 queries),
well within this limit.  If you add many new classes, monitor for silent score drops.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ultralytics import — kept at module level so tests can patch it
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLOWorld as _YOLOWorld
except ImportError:  # pragma: no cover
    _YOLOWorld = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Query map
# ---------------------------------------------------------------------------
QUERY_MAP: dict[str, list[str]] = {
    "Exposed_reinforcement": [
        "bare metal rod",
        "protruding steel bar",
        "rebar sticking out of concrete",
        "metal rod in broken concrete",
        "corroded steel bar",
    ],
    "RustStain": [
        "brown stain on concrete",
        "orange streak on wall",
        "rust mark on surface",
        "iron stain on cement",
        "reddish discoloration",
    ],
    "Scaling": [
        "peeling concrete surface",
        "flaking wall layer",
        "surface layer coming off",
        "deteriorating concrete top",
        "rough eroded surface",
    ],
    "Spalling": [
        "broken concrete chunk",
        "missing piece of wall",
        "concrete falling off",
        "deep chip in concrete",
        "hollow area in wall",
    ],
    "Crack": [
        "thin line in concrete",
        "fracture in wall",
        "hairline crack",
        "vertical crack in wall",
        "horizontal crack in concrete",
    ],
    "Efflorescence": [
        "white powder on wall",
        "white crust on concrete",
        "salt deposit on surface",
        "chalky white stain",
        "mineral deposit on brick",
    ],
}


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single detection returned by MultiQueryYOLOWorld.predict()."""

    label: str          # canonical class name from QUERY_MAP
    confidence: float   # post-NMS confidence score (0–1)
    bbox: list[float]   # [x1, y1, x2, y2] in pixels
    query_matched: str  # the sub-query string that produced this detection


# ---------------------------------------------------------------------------
# Internal helpers (module-level for testability)
# ---------------------------------------------------------------------------

def _build_reverse_map(query_map: dict[str, list[str]]) -> dict[str, str]:
    """Return ``{sub_query: canonical_class}`` built from *query_map*.

    Logs a warning if the same sub-query string appears under two different
    canonical classes (first mapping wins).
    """
    reverse: dict[str, str] = {}
    for canonical, queries in query_map.items():
        for q in queries:
            if q in reverse:
                logger.warning(
                    "Duplicate sub-query %r: already mapped to %r, ignoring %r",
                    q, reverse[q], canonical,
                )
            else:
                reverse[q] = canonical
    return reverse


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_pure(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy NMS for a *single-class* list, sorted by confidence descending."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(_iou(det.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def _nms_torchvision(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """torchvision NMS for a single-class list — preferred on CUDA Jetson devices."""
    import torch
    import torchvision.ops as ops

    boxes  = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
    scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
    keep   = ops.nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep.tolist()]


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """NMS dispatcher: torchvision when available, pure-Python fallback."""
    try:
        return _nms_torchvision(detections, iou_threshold)
    except ImportError:
        return _nms_pure(detections, iou_threshold)


def _nms_per_class(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Apply NMS independently within each canonical class group.

    Boxes from *different* classes are never compared — a Crack and a Spalling
    box that perfectly overlap will both survive.
    """
    groups: dict[str, list[Detection]] = {}
    for det in detections:
        groups.setdefault(det.label, []).append(det)

    kept: list[Detection] = []
    for group in groups.values():
        kept.extend(_nms(group, iou_threshold) if len(group) > 1 else group)
    return kept


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiQueryYOLOWorld:
    """YOLO-World wrapper that runs multiple plain-language sub-queries per class.

    Example
    -------
    >>> detector = MultiQueryYOLOWorld("yolov8s-worldv2.pt", conf=0.25, iou_nms=0.4)
    >>> results = detector.predict(frame_bgr)   # List[Detection]
    """

    def __init__(
        self,
        model_path: str = "yolov8s-worldv2.pt",
        conf: float = 0.25,
        iou_nms: float = 0.4,
        query_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """
        Args:
            model_path: Path to a YOLO-World .pt checkpoint.
            conf:       Minimum confidence threshold; detections below this are
                        discarded before NMS.
            iou_nms:    IoU threshold for per-class NMS (0–1).
            query_map:  Override QUERY_MAP, e.g. to add new defect classes.
                        Defaults to the module-level QUERY_MAP.
        """
        if _YOLOWorld is None:
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            )

        self.conf      = conf
        self.iou_nms   = iou_nms
        self.query_map = query_map or QUERY_MAP

        self._reverse_map  = _build_reverse_map(self.query_map)
        self._all_queries  = [q for qs in self.query_map.values() for q in qs]

        self._model = _YOLOWorld(model_path)
        # Single CLIP text-encoder forward pass — all sub-queries encoded at once
        self._model.set_classes(self._all_queries)

        logger.info(
            "MultiQueryYOLOWorld ready | %d canonical classes | %d sub-queries | %s",
            len(self.query_map), len(self._all_queries), model_path,
        )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame and return merged detections.

        Args:
            frame: BGR ``np.ndarray`` (H, W, 3) — as returned by cv2.imread /
                   cv2.imdecode.  Does NOT modify the array.

        Returns:
            List of :class:`Detection` objects, sorted by confidence descending.
            Boxes from the same class that overlap above ``iou_nms`` are merged;
            boxes from different classes are never merged.
        """
        raw_results = self._model.predict(frame, conf=self.conf, verbose=False)

        raw: list[Detection] = []
        for result in raw_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_idx   = int(box.cls[0])
                query_str = result.names[cls_idx]
                canonical = self._reverse_map.get(query_str)
                if canonical is None:
                    logger.warning("Unrecognised query in result: %r — skipping", query_str)
                    continue
                raw.append(Detection(
                    label         = canonical,
                    confidence    = float(box.conf[0]),
                    bbox          = box.xyxy[0].tolist(),
                    query_matched = query_str,
                ))

        merged = _nms_per_class(raw, self.iou_nms)
        return sorted(merged, key=lambda d: d.confidence, reverse=True)


# ---------------------------------------------------------------------------
# CLI benchmark
# ---------------------------------------------------------------------------

def _benchmark(
    model_path: str,
    frames_dir: str,
    conf: float,
    iou_nms: float,
    n_warmup: int = 3,
) -> None:
    """Measure average predict() latency across all frames in *frames_dir*."""
    import cv2  # local import — only needed for CLI

    detector = MultiQueryYOLOWorld(model_path=model_path, conf=conf, iou_nms=iou_nms)

    frame_paths = sorted(Path(frames_dir).glob("*.jpg")) + \
                  sorted(Path(frames_dir).glob("*.jpeg")) + \
                  sorted(Path(frames_dir).glob("*.png"))

    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]
    print(f"Loaded {len(frames)} frame(s) from {frames_dir}")

    print(f"Warming up with {min(n_warmup, len(frames))} frame(s)…")
    for f in frames[:n_warmup]:
        detector.predict(f)

    times: list[float] = []
    for i, f in enumerate(frames):
        t0   = time.perf_counter()
        dets = detector.predict(f)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  frame {i+1:3d}  {len(dets):2d} det(s)  {elapsed*1000:.1f} ms")

    avg = sum(times) / len(times) * 1000
    print(f"\n{'─'*42}")
    print(f"  frames : {len(times)}")
    print(f"  avg    : {avg:.1f} ms")
    print(f"  min    : {min(times)*1000:.1f} ms")
    print(f"  max    : {max(times)*1000:.1f} ms")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="MultiQueryYOLOWorld")
    ap.add_argument("--benchmark", action="store_true",
                    help="Measure average inference time per frame")
    ap.add_argument("--model",  default="yolov8s-worldv2.pt",
                    help="Path to YOLO-World .pt checkpoint")
    ap.add_argument("--frames", default="data/test_frames",
                    help="Directory of .jpg/.png frames to benchmark on")
    ap.add_argument("--conf",   type=float, default=0.25,
                    help="Confidence threshold (default 0.25)")
    ap.add_argument("--iou",    type=float, default=0.4,
                    help="NMS IoU threshold (default 0.4)")
    args = ap.parse_args()

    if args.benchmark:
        _benchmark(args.model, args.frames, args.conf, args.iou)
    else:
        ap.print_help()
