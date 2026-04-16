"""
DINOv2 Feature Embedder for Hawk-I
------------------------------------
Extracts 768-dim CLS-token embeddings from SAM2-masked defect crops.

Two purposes:
  1. Defect similarity search — nearest neighbour in embedding space.
  2. Confidence re-scoring — if the new defect embedding is far from its
     class centroid (cosine_similarity < 0.45), flag it LOW_CONFIDENCE
     and downgrade severity by one level before saving to DB.

Singleton pattern: load model once via get_embedder().
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ImageNet normalisation constants used by DINOv2
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LOW_SIMILARITY_THRESHOLD: float = 0.45   # below this → flag + downgrade severity
_MIN_CLASS_EXAMPLES: int        = 5      # minimum stored embeddings for centroid


class DINOv2Embedder:
    """
    Singleton DINOv2-base (768-dim) embedder.

    Load facebook/dinov2-base once at startup (via transformers, falling back
    to torch.hub).  Model runs on CUDA (fp16) when available, CPU otherwise.
    """

    def __init__(self) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._use_transformers: bool = True
        # centroid cache: class_name → np.ndarray(768,) or None
        self._centroid_cache: dict[str, Optional[np.ndarray]] = {}

    # ── Lazy model load ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return

        logger.info("DINOv2: loading facebook/dinov2-base…")
        t0 = time.time()

        # Primary: HuggingFace transformers
        try:
            from transformers import AutoModel
            mdl = AutoModel.from_pretrained("facebook/dinov2-base")
            mdl.eval()
            if self.device == "cuda":
                mdl = mdl.half().to(self.device)
            else:
                mdl = mdl.to(self.device)
            self._model = mdl
            self._use_transformers = True
            logger.info(
                "DINOv2 loaded via transformers in %.0fms on %s",
                (time.time() - t0) * 1000, self.device,
            )
            return
        except Exception as e:
            logger.warning("DINOv2: transformers load failed (%s) → trying torch.hub", e)

        # Fallback: torch.hub
        try:
            mdl = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True,
            )
            mdl.eval()
            if self.device == "cuda":
                mdl = mdl.half().to(self.device)
            else:
                mdl = mdl.to(self.device)
            self._model = mdl
            self._use_transformers = False
            logger.info(
                "DINOv2 loaded via torch.hub in %.0fms on %s",
                (time.time() - t0) * 1000, self.device,
            )
        except Exception as e2:
            raise RuntimeError(
                "Cannot load DINOv2-base. Install: pip install transformers"
            ) from e2

    # ── Core embedding ─────────────────────────────────────────────────────

    def embed_crop(
        self,
        frame_rgb: np.ndarray,
        mask: Optional[np.ndarray],
        bbox: list[float],
    ) -> np.ndarray:
        """
        Extract 768-dim CLS embedding from a SAM2-masked defect crop.

        Parameters
        ----------
        frame_rgb : H×W×3 uint8 numpy array in RGB colour order.
        mask      : H×W bool array from SAM2 (zeros out background).
                    Pass None to embed the raw bbox crop without masking.
        bbox      : [x1, y1, x2, y2] pixel coordinates.

        Returns
        -------
        np.ndarray shape (768,) dtype float32.
        """
        import cv2

        self._load_model()
        t0 = time.time()

        h, w = frame_rgb.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, max(x1 + 1, int(bbox[2])))
        y2 = min(h, max(y1 + 1, int(bbox[3])))

        # Zero out background using the SAM2 mask
        if mask is not None and mask.shape == (h, w):
            src = frame_rgb.copy()
            src[~mask] = 0
        else:
            src = frame_rgb

        crop = src[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("DINOv2: empty crop (bbox=%s) → zero embedding", bbox)
            return np.zeros(768, dtype=np.float32)

        # Resize to 224×224 (DINOv2 input size) and ImageNet-normalise
        crop_224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        norm = (crop_224.astype(np.float32) / 255.0 - _MEAN) / _STD   # (224,224,3)

        # Build (1, 3, 224, 224) tensor
        t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
        if self.device == "cuda":
            t = t.half().to(self.device)
        else:
            t = t.float().to(self.device)

        with torch.no_grad():
            if self._use_transformers:
                out = self._model(pixel_values=t)
                cls_tok = out.last_hidden_state[:, 0, :]   # (1, 768)
            else:
                cls_tok = self._model(t)                    # (1, 768) from hub

        emb = cls_tok.squeeze(0).float().cpu().numpy().astype(np.float32)

        elapsed = (time.time() - t0) * 1000
        if elapsed > 150:
            logger.warning("DINOv2: embed_crop %.0fms — exceeds 150ms threshold", elapsed)
        elif elapsed > 80:
            logger.info("DINOv2: embed_crop %.0fms (>80ms soft limit)", elapsed)

        return emb

    # ── Similarity helpers ─────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity ∈ [-1, 1] using pure numpy (no sklearn)."""
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    async def find_similar(
        self,
        embedding: np.ndarray,
        class_name: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Return top_k most visually similar past detections (same class).

        Fetches stored BYTEA embeddings from DB, decodes them, and ranks by
        cosine similarity in Python (no pgvector required).

        Returns list of dicts:
            {id, lat, lon, detected_at, similarity_score, area_cm2}
        """
        from database import pool, CURRENT_TABLE

        if pool is None:
            return []

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT id, lat, lon, detected_at, area_cm2, embedding
                      FROM {CURRENT_TABLE}
                     WHERE class_name = $1
                       AND embedding  IS NOT NULL
                     ORDER BY detected_at DESC
                     LIMIT 500
                    """,
                    class_name,
                )
        except Exception as exc:
            logger.warning("DINOv2 find_similar: DB query failed — %s", exc)
            return []

        scored: list[dict] = []
        for row in rows:
            try:
                stored = np.frombuffer(bytes(row["embedding"]), dtype=np.float32)
                if stored.shape != (768,):
                    continue
                sim = DINOv2Embedder.cosine_similarity(embedding, stored)
                scored.append(
                    {
                        "id":               row["id"],
                        "lat":              float(row["lat"]),
                        "lon":              float(row["lon"]),
                        "detected_at":      str(row["detected_at"]),
                        "similarity_score": round(sim, 4),
                        "area_cm2":         float(row["area_cm2"] or 0.0),
                    }
                )
            except Exception:
                continue

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    async def get_class_centroid(
        self, class_name: str
    ) -> Optional[np.ndarray]:
        """
        Return the mean embedding for all stored detections of class_name.

        Cached in memory; call invalidate_centroid() after saving a new row.
        Returns None when fewer than _MIN_CLASS_EXAMPLES embeddings exist
        (not enough data to be reliable).
        """
        if class_name in self._centroid_cache:
            return self._centroid_cache[class_name]

        from database import pool, CURRENT_TABLE

        if pool is None:
            self._centroid_cache[class_name] = None
            return None

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT embedding
                      FROM {CURRENT_TABLE}
                     WHERE class_name = $1
                       AND embedding  IS NOT NULL
                    """,
                    class_name,
                )
        except Exception as exc:
            logger.warning("DINOv2 get_class_centroid: DB query failed — %s", exc)
            self._centroid_cache[class_name] = None
            return None

        valid: list[np.ndarray] = []
        for row in rows:
            try:
                emb = np.frombuffer(bytes(row["embedding"]), dtype=np.float32)
                if emb.shape == (768,):
                    valid.append(emb)
            except Exception:
                continue

        if len(valid) < _MIN_CLASS_EXAMPLES:
            self._centroid_cache[class_name] = None
            return None

        centroid = np.mean(valid, axis=0).astype(np.float32)
        self._centroid_cache[class_name] = centroid
        return centroid

    def invalidate_centroid(self, class_name: str) -> None:
        """Discard cached centroid so the next call re-queries the DB."""
        self._centroid_cache.pop(class_name, None)


# ── Module-level singleton ─────────────────────────────────────────────────────

_embedder: Optional[DINOv2Embedder] = None


def get_embedder() -> DINOv2Embedder:
    """Return the shared DINOv2Embedder instance (created on first call)."""
    global _embedder
    if _embedder is None:
        _embedder = DINOv2Embedder()
    return _embedder
