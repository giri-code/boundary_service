import threading
import cv2
import numpy as np
import requests
from .base import BaseStorageProvider
from .local_provider import LocalStorageProvider
from .s3_provider import S3StorageProvider
from .gcs_provider import GCSStorageProvider
from ..config import settings
from ..utils.logger import logger


class HTTPStorageProvider(BaseStorageProvider):
    """HTTP / HTTPS Presigned URL Storage Provider."""

    def read_image(self, path_or_uri: str) -> np.ndarray:
        try:
            response = requests.get(path_or_uri, timeout=15, stream=True)
            response.raise_for_status()

            # Guard: Content-Length before buffering (FIX PERF-3 – HTTP variant)
            content_length = int(response.headers.get("Content-Length", 0))
            max_bytes = settings.MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024
            if content_length and content_length > max_bytes:
                raise ValueError(
                    f"Remote image size ({content_length / (1024*1024):.1f} MB) "
                    f"exceeds the configured limit ({settings.MAX_IMAGE_FILE_SIZE_MB} MB)."
                )

            image_bytes = response.content
        except requests.RequestException as exc:
            raise IOError(f"Failed to fetch image from URL '{path_or_uri}': {exc}") from exc

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image from HTTP URL: {path_or_uri}")

        # Guard pixel limit
        h, w = image.shape[:2]
        if h * w > settings.MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image dimensions ({w}×{h} = {h * w} px) exceed "
                f"the pixel limit ({settings.MAX_IMAGE_PIXELS} px)."
            )

        return image

    def exists(self, path_or_uri: str) -> bool:
        try:
            r = requests.head(path_or_uri, timeout=5)
            return r.status_code == 200
        except Exception:
            return False


class StorageProviderFactory:
    """Thread-safe factory to auto-resolve the correct storage provider from a URI.

    Providers are singletons — created once and reused across all requests.
    A threading.Lock guards concurrent first-initialisation (FIX PERF-1).
    """

    _instances: dict = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_provider(cls, path_or_uri: str = None) -> BaseStorageProvider:
        """Resolve provider from URI scheme, then fall back to global STORAGE_BACKEND config."""
        uri = (path_or_uri or "").strip()

        # URI-scheme based routing
        if uri.startswith("s3://"):
            key = "s3"
        elif uri.startswith("gs://"):
            key = "gcs"
        elif uri.startswith("http://") or uri.startswith("https://"):
            key = "http"
        else:
            # Use globally configured backend
            backend = settings.STORAGE_BACKEND.lower()
            if backend == "s3":
                key = "s3"
            elif backend == "gcs":
                key = "gcs"
            else:
                key = "local"

        # Fast path — no lock needed once populated
        if key in cls._instances:
            return cls._instances[key]

        # Slow path — thread-safe lazy initialisation (double-checked locking)
        with cls._lock:
            if key in cls._instances:
                return cls._instances[key]

            if key == "s3":
                cls._instances[key] = S3StorageProvider()
            elif key == "gcs":
                cls._instances[key] = GCSStorageProvider()
            elif key == "http":
                cls._instances[key] = HTTPStorageProvider()
            else:
                cls._instances[key] = LocalStorageProvider()

        return cls._instances[key]

    @classmethod
    def read_image(cls, path_or_uri: str) -> np.ndarray:
        """Convenience wrapper: resolve provider and read image in one call."""
        provider = cls.get_provider(path_or_uri)
        return provider.read_image(path_or_uri)
