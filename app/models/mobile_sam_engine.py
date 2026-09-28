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
        """Decode a click mask from a cached embedding.

        WIN1 (pick-before-upscale): the mask decoder natively emits
        ``low_res_masks (1, 3, 256, 256)`` + ``iou_predictions (1, 3)``.
        The vendored ``SamPredictor.predict()`` upscales ALL 3 candidates to
        full resolution in float32 (``postprocess_masks``) and we previously
        discarded 2 of them. Here we pick the winner at 256x256 and upscale
        only that single mask — ~66% less transient mask memory — without
        touching ``vendor/MobileSAM/*`` (we only *call* the shared read-only
        model from a request-local predictor).
        """
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
                device=self._sam_model.device,
                dtype=torch.float32,
            )
            predictor.features = features
            predictor.is_image_set = True
            predictor.original_size = embedding_dict["original_size"]
            predictor.input_size = embedding_dict["input_size"]

            px, py = point
            input_point = np.array([[px, py]])
            input_label = np.array([1])  # 1 = foreground click prompt

            # ── Same prompt transform as SamPredictor.predict() ──
            coords = predictor.transform.apply_coords(
                input_point, predictor.original_size
            )
            coords_torch = torch.as_tensor(
                coords, dtype=torch.float, device=self._sam_model.device
            )
            labels_torch = torch.as_tensor(
                input_label, dtype=torch.int, device=self._sam_model.device
            )
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

            # ── Lightweight decoder at 256x256 (no upscale yet) ──
            sparse_embeddings, dense_embeddings = self._sam_model.prompt_encoder(
                points=(coords_torch, labels_torch),
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = self._sam_model.mask_decoder(
                image_embeddings=predictor.features,
                image_pe=self._sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            scores = iou_predictions[0].detach().cpu().numpy()
            # Low-res areas are only a proxy for the legacy full-res area
            # ordering used by `level`. Score-based picks (the common path)
            # are exact; level ordering matches in practice but can flip when
            # two candidates have near-identical areas post-interpolation.
            low_res_bin = (
                low_res_masks[0] > self._sam_model.mask_threshold
            ).to(torch.int32)
            low_res_areas = low_res_bin.sum(dim=(1, 2)).detach().cpu().tolist()
            sorted_by_area = sorted(
                range(len(low_res_areas)), key=lambda i: low_res_areas[i]
            )

            if level is not None and 0 <= level < len(sorted_by_area):
                best_idx = sorted_by_area[level]
            else:
                best_idx = int(np.argmax(scores))

            # ── Upscale ONLY the winner: 1x1x256x256 → HxW ──
            winner = low_res_masks[:, best_idx : best_idx + 1, :, :]
            upscaled = self._sam_model.postprocess_masks(
                winner, predictor.input_size, predictor.original_size
            )
            best_mask = (
                (upscaled > self._sam_model.mask_threshold)[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
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
