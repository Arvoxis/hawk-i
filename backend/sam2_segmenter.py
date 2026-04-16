"""
SAM 2 Segmentation Module for ZeroDefect / Hawki
-------------------------------------------------
Takes YOLO/GDINO bounding boxes + drone frame → returns pixel-precise masks
and real-world defect area in cm².

Usage:
    from sam2_segmenter import SAM2Segmenter
    segmenter = SAM2Segmenter("models/sam2.1_hiera_small.pt")
    results = segmenter.segment_detections(frame, detections, altitude_m=12.0)
"""

import os
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera intrinsics — configurable via .env for different drone hardware.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = str(_PROJECT_ROOT / "models" / "sam2.1_hiera_small.pt")

DEFAULT_FOCAL_LENGTH_MM = float(os.getenv("CAMERA_FOCAL_MM",        "3.67"))
DEFAULT_SENSOR_WIDTH_MM = float(os.getenv("CAMERA_SENSOR_WIDTH_MM", "6.287"))
DEFAULT_IMAGE_WIDTH_PX  = int(os.getenv("CAMERA_IMAGE_WIDTH_PX",    "1280"))


class SAM2Segmenter:
    """Wraps SAM 2 image predictor for infrastructure defect segmentation."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: Optional[str] = None,
        focal_length_mm: float = DEFAULT_FOCAL_LENGTH_MM,
        sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self._predictor = None
        self._checkpoint_path = checkpoint_path
        self._model_cfg = model_cfg
        logger.info(f"SAM2Segmenter will use device: {self.device}")

    # ------------------------------------------------------------------
    # Lazy-load so the server starts fast; model loads on first request
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._predictor is not None:
            return

        logger.info("Loading SAM 2 model (first request)...")

        # Enable cuDNN auto-tuner — finds the fastest conv algorithm for the
        # fixed input size (image encoder) and caches it after the first run.
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on Ampere

        try:
            # Primary: load from local checkpoint
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self._model_cfg, self._checkpoint_path, device=self.device)
            self._predictor = SAM2ImagePredictor(model)

        except Exception as e:
            logger.warning(f"Local checkpoint load failed ({e}), trying HuggingFace...")
            try:
                # Fallback: load from HuggingFace
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                self._predictor = SAM2ImagePredictor.from_pretrained(
                    "facebook/sam2.1-hiera-small"
                )
            except Exception as e2:
                logger.error(f"HuggingFace load also failed: {e2}")
                raise RuntimeError(
                    "Could not load SAM 2. Make sure you have installed sam2:\n"
                    "  pip install sam2\n"
                    "  OR git clone https://github.com/facebookresearch/sam2 && cd sam2 && pip install -e .\n"
                    "And downloaded weights:\n"
                    "  python -c \"from sam2.build_sam import build_sam2; ...\"\n"
                    "  OR place checkpoint at models/sam2.1_hiera_small.pt"
                ) from e2

        logger.info("SAM 2 model loaded successfully.")

    # ------------------------------------------------------------------
    # Core: segment a single bounding box in the frame
    # ------------------------------------------------------------------
    def _autocast_ctx(self):
        """Return the correct autocast context for the current device.

        torch.autocast on CPU only supports bfloat16; float32 raises
        RuntimeError on recent PyTorch versions.  On CPU we use a no-op
        context manager so SAM2 runs without autocast overhead.
        """
        import contextlib
        device_type = self.device.split(":")[0]  # "cuda" or "cpu"
        if device_type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def segment_box(self, frame_rgb: np.ndarray, box_xyxy: list[float]) -> dict:
        """
        Segment a single bounding box region.

        Args:
            frame_rgb: HxWx3 numpy array (RGB)
            box_xyxy:  [x1, y1, x2, y2] bounding box

        Returns:
            dict with keys:
                mask:      HxW boolean numpy array
                score:     float confidence
                area_px:   int pixel count of defect
        """
        self._load_model()

        with torch.inference_mode(), self._autocast_ctx():
            self._predictor.set_image(frame_rgb)

            box_np = np.array(box_xyxy, dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=box_np,
                multimask_output=False,     # single best mask
            )

        mask = masks[0]  # shape: (H, W) boolean
        score = float(scores[0])
        area_px = int(mask.sum())

        return {
            "mask": mask,
            "score": score,
            "area_px": area_px,
        }

    # ------------------------------------------------------------------
    # Convert pixel area → real-world cm² using drone altitude
    # ------------------------------------------------------------------
    def px_to_cm2(
        self,
        area_px: int,
        altitude_m: float,
        image_width_px: int = DEFAULT_IMAGE_WIDTH_PX,
    ) -> float:
        """
        Convert pixel area to real-world area in cm².

        Uses the pinhole camera model:
            GSD (cm/px) = (altitude_cm × sensor_width_mm) / (focal_length_mm × image_width_px × 10)
            area_cm2 = area_px × GSD²

        Args:
            area_px:        number of pixels in the mask
            altitude_m:     drone altitude above ground in meters
            image_width_px: image width in pixels

        Returns:
            float: area in cm²
        """
        altitude_cm = altitude_m * 100.0

        # Ground Sampling Distance: how many cm each pixel represents
        gsd_cm_per_px = (
            altitude_cm * self.sensor_width_mm
        ) / (
            self.focal_length_mm * image_width_px * 10.0
        )

        area_cm2 = area_px * (gsd_cm_per_px ** 2)
        return round(area_cm2, 2)

    # ------------------------------------------------------------------
    # High-level: segment all detections in a frame
    # ------------------------------------------------------------------
    def segment_detections(
        self,
        frame_rgb: np.ndarray,
        detections: list[dict],
        altitude_m: float = 10.0,
    ) -> list[dict]:
        """
        Segment all detections from YOLO/GDINO in a single frame.

        Args:
            frame_rgb:  HxWx3 numpy array (RGB)
            detections: list of dicts, each with a "box" key [x1,y1,x2,y2]
                        (from yolo_detections or gdino_detections in your JSON)
            altitude_m: drone altitude from GPS payload

        Returns:
            list of dicts, one per detection:
                {
                    "mask": np.ndarray (H,W bool),
                    "score": float,
                    "area_px": int,
                    "area_cm2": float,
                    "box": [x1,y1,x2,y2],
                    "class": str (if present),
                    "conf": float (if present),
                }
        """
        self._load_model()

        results = []
        h, w = frame_rgb.shape[:2]

        # Set image once, predict for each box
        with torch.inference_mode(), self._autocast_ctx():
            self._predictor.set_image(frame_rgb)

            for det in detections:
                box = det.get("box") or det.get("bbox")
                if not box:
                    logger.warning(f"Detection missing 'box' key: {det}")
                    continue

                try:
                    box_np = np.array(box, dtype=np.float32)
                    masks, scores, _ = self._predictor.predict(
                        box=box_np,
                        multimask_output=False,
                    )

                    mask = masks[0]
                    score = float(scores[0])
                    area_px = int(mask.sum())
                    area_cm2 = self.px_to_cm2(area_px, altitude_m, image_width_px=w)

                    results.append({
                        "mask": mask,
                        "score": score,
                        "area_px": area_px,
                        "area_cm2": area_cm2,
                        "box": box,
                        # pass through original detection info
                        "class": det.get("class") or det.get("phrase", "unknown"),
                        "conf": det.get("conf", 0.0),
                    })

                except Exception as e:
                    logger.error(f"SAM2 failed on box {box}: {e}")
                    continue

        return results

    # ------------------------------------------------------------------
    # Health check — call at startup to fail fast if SAM2 is broken
    # ------------------------------------------------------------------
    def sam2_health_check(self) -> bool:
        """
        Load the model and run a dummy 224×224 inference to verify it works.

        Returns True if healthy, False if any step fails.
        Call once at backend startup; log result.
        """
        try:
            self._load_model()
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            with torch.inference_mode(), self._autocast_ctx():
                self._predictor.set_image(dummy)
                box = np.array([56.0, 56.0, 168.0, 168.0], dtype=np.float32)
                masks, scores, _ = self._predictor.predict(
                    box=box, multimask_output=False
                )
            logger.info(
                "SAM2 health check PASSED — score=%.3f", float(scores[0])
            )
            return True
        except Exception as exc:
            logger.error("SAM2 health check FAILED: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Utility: draw mask overlay on frame for dashboard / annotated image
    # ------------------------------------------------------------------
    @staticmethod
    def draw_mask_overlay(
        frame_rgb: np.ndarray,
        mask: np.ndarray,
        color: tuple = (255, 0, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay a semi-transparent mask on the frame.

        Args:
            frame_rgb: HxWx3 numpy array
            mask:      HxW boolean array
            color:     RGB tuple for mask color
            alpha:     transparency (0=invisible, 1=opaque)

        Returns:
            HxWx3 numpy array with mask overlaid
        """
        overlay = frame_rgb.copy()
        overlay[mask] = (
            np.array(color) * alpha + overlay[mask] * (1 - alpha)
        ).astype(np.uint8)
        return overlay  