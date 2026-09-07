import threading
from typing import Dict
from .base import BaseSegmentationEngine
from .mobile_sam_engine import MobileSAMEngine
from .sam2_engine import SAM2Engine
from .opencv_engine import OpenCVEngine
from ..config import settings


class SegmentationModelFactory:
    """Thread-safe factory to retrieve or switch segmentation model engine implementations.

    Engines are singletons — created once per provider name and reused across requests.
    A threading.Lock guards concurrent first-initialization to avoid race conditions (PERF-1).
    """

    _instances: Dict[str, BaseSegmentationEngine] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_engine(cls, provider_name: str = None) -> BaseSegmentationEngine:
        """Return the singleton engine for the given provider name.

        Args:
            provider_name: One of 'mobile_sam', 'sam2', 'opencv', or None to use default.

        Returns:
            A concrete BaseSegmentationEngine implementation.
        """
        provider = (provider_name or settings.SEGMENTATION_MODEL_PROVIDER).lower()

        # Fast path: no lock needed if already initialised
        if provider in cls._instances:
            return cls._instances[provider]

        # Slow path: acquire lock for thread-safe first creation (FIX PERF-1)
        with cls._lock:
            # Double-checked locking pattern — re-check inside the lock
            if provider in cls._instances:
                return cls._instances[provider]

            if provider in ("mobile_sam", "mobilesam", "sam"):
                cls._instances[provider] = MobileSAMEngine()
            elif provider in ("sam2", "sam_2"):
                cls._instances[provider] = SAM2Engine()
            elif provider in ("opencv", "grabcut", "cv"):
                cls._instances[provider] = OpenCVEngine()
            else:
                # Unknown provider — default to MobileSAM
                cls._instances[provider] = MobileSAMEngine()

        return cls._instances[provider]
