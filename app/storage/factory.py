import threading
import numpy as np
from .base import BaseStorageProvider
from .s3_provider import S3StorageProvider


class StorageProviderFactory:
    """S3-only storage factory (gallery storage is S3-URI-based end to end).

    `http(s)://` image paths are rejected — no SSRF-capable fetcher exists in
    this service. Thread-safe singleton with double-checked locking.
    """

    _instances: dict = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_provider(cls, path_or_uri: str = None) -> BaseStorageProvider:
        """Return the singleton S3StorageProvider, rejecting http(s) URIs."""
        uri = (path_or_uri or "").strip()

        if uri.startswith("http://") or uri.startswith("https://"):
            raise ValueError(
                f"HTTP(S) image URLs are not supported: '{uri}'. "
                "Use an s3:// URI or bucket-relative key."
            )

        # Fast path — no lock needed once populated
        if "s3" in cls._instances:
            return cls._instances["s3"]

        # Slow path — thread-safe lazy initialisation (double-checked locking)
        with cls._lock:
            if "s3" in cls._instances:
                return cls._instances["s3"]

            cls._instances["s3"] = S3StorageProvider()

        return cls._instances["s3"]

    @classmethod
    def read_image(cls, path_or_uri: str) -> np.ndarray:
        """Convenience wrapper: resolve provider and read image in one call."""
        provider = cls.get_provider(path_or_uri)
        return provider.read_image(path_or_uri)
