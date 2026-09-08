from typing import Tuple, Optional
import cv2
import numpy as np
from .base import BaseSegmentationEngine
from ..config import settings


class OpenCVEngine(BaseSegmentationEngine):
    """High-Performance Computer Vision Boundary Engine.

    Uses GrabCut (primary) with Adaptive FloodFill fallback.
    Supports Region-of-Interest cropping for high-resolution images to
    constrain computation to the area around the click point.
    """

    # FIX MEM-3: pre-allocate GrabCut model arrays as class-level constants
    # These are overwritten in-place by cv2.grabCut so they cannot be truly
    # shared across concurrent calls — we keep them here as a template and
    # copy-allocate per call, but the pattern is documented for future GPU work.
    _BGDMODEL_SHAPE = (1, 65)
    _FGDMODEL_SHAPE = (1, 65)

    def __init__(self):
        super().__init__(name="OpenCV-GrabCut")

    def predict_mask(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        cache_key: Optional[str] = None,
        level: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        px = max(0, min(w - 1, point[0]))
        py = max(0, min(h - 1, point[1]))

        # Performance: crop a window around the click point for large images
        if settings.ENABLE_ROI_CROPPING and (
            w > settings.MAX_ROI_DIMENSION or h > settings.MAX_ROI_DIMENSION
        ):
            roi_half = settings.MAX_ROI_DIMENSION // 2
            x1 = max(0, px - roi_half)
            y1 = max(0, py - roi_half)
            x2 = min(w, px + roi_half)
            y2 = min(h, py + roi_half)

            sub_img = image[y1:y2, x1:x2]
            sub_mask, confidence = self._segment_roi(sub_img, (px - x1, py - y1))

            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[y1:y2, x1:x2] = sub_mask
            return full_mask, confidence

        return self._segment_roi(image, (px, py))

    def _segment_roi(
        self, image: np.ndarray, point: Tuple[int, int]
    ) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        px, py = point

        # ── Method 1: GrabCut ─────────────────────────────────────────────────
        try:
            mask = np.zeros((h, w), np.uint8)
            # GrabCut writes into these arrays — must be freshly allocated per call
            bgd_model = np.zeros(self._BGDMODEL_SHAPE, np.float64)
            fgd_model = np.zeros(self._FGDMODEL_SHAPE, np.float64)

            margin = min(w, h) // 4
            rect = (
                max(0, px - margin),
                max(0, py - margin),
                max(1, min(w - 1, px + margin) - max(0, px - margin)),
                max(1, min(h - 1, py + margin) - max(0, py - margin)),
            )

            if rect[2] > 5 and rect[3] > 5:
                cv2.grabCut(
                    image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT
                )
                cv2.circle(mask, (px, py), 5, cv2.GC_FGD, -1)
                cv2.grabCut(
                    image, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK
                )

                binary_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
                if np.any(binary_mask) and binary_mask[py, px]:
                    return binary_mask, 0.88

        except Exception:
            pass  # fall through to FloodFill

        # ── Method 2: Adaptive FloodFill (magic wand) ─────────────────────────
        # FIX PERF-2: FLOODFILL_MASK_ONLY does NOT modify image — no copy needed
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        lo_diff = (20, 20, 20)
        up_diff = (20, 20, 20)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

        cv2.floodFill(
            image, flood_mask, (px, py), (255, 255, 255), lo_diff, up_diff, flags
        )

        binary_mask = flood_mask[1:-1, 1:-1] == 255
        if not np.any(binary_mask):
            # Last resort: small circle around click point
            binary_mask = np.zeros((h, w), dtype=bool)
            cv2.circle(binary_mask.view(np.uint8), (px, py), 20, 1, -1)

        return binary_mask, 0.75
