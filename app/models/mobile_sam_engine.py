import os
from typing import Tuple, Optional
import numpy as np

from .base import BaseSegmentationEngine
from ..config import settings
from ..utils.logger import logger
from ..utils.memory import force_garbage_collection


class MobileSAMEngine(BaseSegmentationEngine):
    """MobileSAM (Segment Anything Model) Engine.

    Features:
    - Lazy model loading: weights are loaded on first prediction call.
    - Embedding caching: image embeddings are cached by cache_key (e.g. file path)
      so repeated clicks on the same image skip the expensive encoder step.
    - torch.inference_mode: disables autograd to reduce memory and speed up inference.
    - Graceful fallback: if weights are missing or the mobile_sam package is absent,
      the engine delegates to the shared OpenCV singleton (not a fresh instance).
    """

    def __init__(self, checkpoint_path: str = None):
        super().__init__(name="MobileSAM")
        self.checkpoint_path = checkpoint_path or settings.MOBILE_SAM_CHECKPOINT
        self._predictor = None
        self._sam_model = None
        self._load_failed: bool = False  # guard against retry storms on broken weights

    def _load_model(self) -> None:
        """Load MobileSAM weights. Raises RuntimeError on failure."""
        if self._predictor is not None or self._load_failed:
            return

        try:
            import torch
            from mobile_sam import sam_model_registry, SamPredictor

            device = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                    else "cpu"
                )
            )

            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(
                    f"MobileSAM checkpoint not found at: {self.checkpoint_path}. "
                    "Ensure 'python scripts/download_weights.py' was run during setup."
                )

            logger.info(
                f"Loading MobileSAM weights from '{self.checkpoint_path}' on device '{device}'"
            )
            self._sam_model = sam_model_registry["vit_t"](checkpoint=self.checkpoint_path)
            self._sam_model.to(device=device)
            self._sam_model.eval()
            self._predictor = SamPredictor(self._sam_model)

        except Exception as exc:
            self._predictor = None
            self._load_failed = True  # prevent repeated re-try on every request
            logger.warning(
                f"MobileSAM load failed: {exc}. "
                "All requests will fall back to OpenCV engine until restart."
            )
            raise RuntimeError(f"Could not load MobileSAM model: {exc}") from exc

    def predict_mask(
        self, image: np.ndarray, point: Tuple[int, int], cache_key: Optional[str] = None, level: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        # FIX BUG-4: use the factory singleton for fallback, not a new instance per request
        if self._load_failed:
            return self._opencv_fallback(image, point, cache_key, level)

        try:
            self._load_model()
        except RuntimeError:
            return self._opencv_fallback(image, point, cache_key, level)

        px, py = point
        input_point = np.array([[px, py]])
        input_label = np.array([1])  # 1 = foreground click prompt

        import torch

        with torch.inference_mode():
            if cache_key and cache_key in self._embedding_cache:
                cached = self._embedding_cache[cache_key]
                self._predictor.features = cached["features"]
                self._predictor.is_image_set = True
                self._predictor.original_size = cached["original_size"]
                self._predictor.input_size = cached["input_size"]
            else:
                import cv2
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self._predictor.set_image(rgb_image)

                if cache_key:
                    if len(self._embedding_cache) >= settings.EMBEDDING_CACHE_SIZE:
                        oldest_key = next(iter(self._embedding_cache))
                        del self._embedding_cache[oldest_key]
                        force_garbage_collection()

                    self._embedding_cache[cache_key] = {
                        "features": self._predictor.features,
                        "original_size": self._predictor.original_size,
                        "input_size": self._predictor.input_size,
                    }

            masks, scores, _ = self._predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        # Sort masks by area: index 0 = smallest (fine/detailed), 2 = largest (coarse/broad)
        # This ensures Fine level reliably picks the tightest boundary (e.g. glasses, not the face)
        mask_areas = [int(m.sum()) for m in masks]
        sorted_by_area = sorted(range(len(masks)), key=lambda i: mask_areas[i])

        if level is not None and 0 <= level < len(sorted_by_area):
            # Fine(0) → smallest area, Medium(1) → mid area, Coarse(2) → largest area
            best_idx = sorted_by_area[level]
        else:
            best_idx = int(np.argmax(scores))

        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])

        return best_mask, best_score

    def _opencv_fallback(
        self, image: np.ndarray, point: Tuple[int, int], cache_key: Optional[str], level: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """Delegate to the shared OpenCV singleton — not a new instance per call."""
        from ..models.factory import SegmentationModelFactory
        engine = SegmentationModelFactory.get_engine("opencv")
        return engine.predict_mask(image, point, cache_key=cache_key, level=level)
