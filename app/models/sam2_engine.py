from typing import Tuple, Optional
import numpy as np
from .base import BaseSegmentationEngine
from ..utils.logger import logger


class SAM2Engine(BaseSegmentationEngine):
    """Segment Anything 2 (SAM 2) Engine.

    When SAM 2 weights / packages are available this engine handles prediction.
    Until then it logs a warning and delegates to the OpenCV fallback engine —
    importantly, any unexpected exception is re-raised rather than silently swallowed.
    """

    def __init__(self):
        super().__init__(name="SAM2")

    def predict_mask(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        cache_key: Optional[str] = None,
        level: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        # FIX BUG-3: distinguish ImportError (package missing) from NotImplementedError
        # (not-yet-wired) so the fallback is intentional and always logged.
        try:
            import sam2  # noqa: F401
        except ImportError:
            logger.warning(
                "SAM 2 package not installed. Falling back to OpenCV engine. "
                "Install 'sam2' to enable SAM 2 segmentation."
            )
            return self._opencv_fallback(image, point, cache_key, level)

        # SAM 2 integration wired but weights / config not yet connected
        logger.warning(
            "SAM 2 package is installed but inference is not yet configured. "
            "Falling back to OpenCV engine."
        )
        return self._opencv_fallback(image, point, cache_key, level)

    def _opencv_fallback(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        cache_key: Optional[str],
        level: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        # FIX BUG-4 (shared pattern): use the factory singleton rather than a new instance
        from ..models.factory import SegmentationModelFactory

        engine = SegmentationModelFactory.get_engine("opencv")
        return engine.predict_mask(image, point, cache_key=cache_key, level=level)
