import os
import threading
from typing import Tuple, Optional
import numpy as np

from .base import BaseSegmentationEngine
from ..config import settings
from ..utils.logger import logger


class MobileSAMEngine(BaseSegmentationEngine):
    """MobileSAM (Segment Anything Model) Engine.

    Features:
    - Lock-free concurrent inference: uses request-local SamPredictor instances over
      shared read-only weights (_sam_model) under torch.inference_mode.
    - Thread-safe lazy initialization: weights are loaded once behind double-checked locking.
    - torch.inference_mode: disables autograd to reduce memory and speed up inference.
    - Fail-loud: startup pre-warm (lifespan) guarantees weights exist; a load
      failure raises instead of silently serving another model's masks.
    """

    def __init__(self, checkpoint_path: str = None):
        super().__init__(name="MobileSAM")
        self.checkpoint_path = checkpoint_path or settings.MOBILE_SAM_CHECKPOINT
        self._sam_model = None
        self._load_failed: bool = False  # guard against retry storms on broken weights
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        """Load MobileSAM weights. Raises RuntimeError on failure."""
        if self._sam_model is not None or self._load_failed:
            return

        with self._lock:
            if self._sam_model is not None or self._load_failed:
                return

            try:
                import torch
                from mobile_sam import sam_model_registry  # type: ignore

                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else (
                        "mps"
                        if hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
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
                self._sam_model = sam_model_registry["vit_t"](
                    checkpoint=self.checkpoint_path
                )
                self._sam_model.to(device=device)
                self._sam_model.eval()

            except Exception as exc:
                self._sam_model = None
                self._load_failed = True  # prevent repeated re-try on every request
                logger.warning(
                    f"MobileSAM load failed: {exc}. "
                    "Service requires restart with valid weights."
                )
                raise RuntimeError(f"Could not load MobileSAM model: {exc}") from exc

    def encode_image(self, image: np.ndarray) -> dict:
        if self._load_failed:
            raise RuntimeError("MobileSAM load failed, cannot encode image.")
        self._load_model()

        import torch
        import cv2
        from mobile_sam import SamPredictor  # type: ignore

        predictor = SamPredictor(self._sam_model)
        with torch.inference_mode():
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb_image)
            features_np = predictor.features.cpu().numpy()
            orig_size = predictor.original_size
            inp_size = predictor.input_size
            return {
                "features": features_np,
                "original_size": orig_size,
                "input_size": inp_size,
            }

    def predict_from_embedding(
        self, embedding_dict: dict, point: Tuple[int, int], level: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        if self._load_failed:
            raise RuntimeError("MobileSAM load failed, cannot predict from embedding.")
        self._load_model()

        import torch
        from mobile_sam import SamPredictor  # type: ignore

        raw_features = embedding_dict["features"]
        if not getattr(raw_features, "flags", None) or not raw_features.flags.writeable:
            raw_features = raw_features.copy()

        predictor = SamPredictor(self._sam_model)
        with torch.inference_mode():
            features = torch.from_numpy(raw_features).to(
                self._sam_model.device
            )
            predictor.features = features
            predictor.is_image_set = True
            predictor.original_size = embedding_dict["original_size"]
            predictor.input_size = embedding_dict["input_size"]

            px, py = point
            input_point = np.array([[px, py]])
            input_label = np.array([1])  # 1 = foreground click prompt

            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        mask_areas = [int(m.sum()) for m in masks]
        sorted_by_area = sorted(range(len(masks)), key=lambda i: mask_areas[i])

        if level is not None and 0 <= level < len(sorted_by_area):
            best_idx = sorted_by_area[level]
        else:
            best_idx = int(np.argmax(scores))

        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])
        return best_mask, best_score

    def predict_mask(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        cache_key: Optional[str] = None,
        level: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        # FIX BUG-4: MobileSAM-only — never fall back to another model.
        # Lifespan pre-warm guarantees weights; reaching here means a genuine
        # load failure, which must surface as an error, not a foreign mask.
        if self._load_failed:
            raise RuntimeError("MobileSAM weights failed to load at startup.")

        try:
            self._load_model()
        except RuntimeError as exc:
            raise RuntimeError(f"MobileSAM unavailable: {exc}") from exc

        embedding_dict = None
        if cache_key and settings.ENABLE_EMBEDDING_CACHE:
            with self._lock:
                embedding_dict = self._embedding_cache.get(cache_key)

        if embedding_dict is None:
            embedding_dict = self.encode_image(image)
            if cache_key and settings.ENABLE_EMBEDDING_CACHE:
                with self._lock:
                    if len(self._embedding_cache) >= settings.EMBEDDING_CACHE_SIZE:
                        oldest_key = next(iter(self._embedding_cache))
                        del self._embedding_cache[oldest_key]
                    self._embedding_cache[cache_key] = embedding_dict

        return self.predict_from_embedding(embedding_dict, point, level)
