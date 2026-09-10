from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseSegmentationEngine(ABC):
    """Abstract Base Class for Object Segmentation Model Engines."""

    def __init__(self, name: str):
        self.name = name
        self._embedding_cache: Dict[str, Any] = {}

    @abstractmethod
    def predict_mask(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        cache_key: Optional[str] = None,
        level: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Predicts binary mask (2D boolean array) and returns (binary_mask, confidence_score).

        Args:
            image: OpenCV BGR/RGB numpy array (H, W, 3)
            point: (x, y) tuple representing pixel coordinates of user click
            cache_key: Optional string identifier (e.g. image path) for caching image embeddings

        Returns:
            Tuple of (binary_mask: np.ndarray [H, W], confidence_score: float)
        """

    def clear_cache(self):
        """Clears cached image embeddings in a thread-safe manner."""
        if hasattr(self, "_lock"):
            with self._lock:
                self._embedding_cache.clear()
        else:
            self._embedding_cache.clear()
