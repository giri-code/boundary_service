from abc import ABC, abstractmethod
import numpy as np


class BaseStorageProvider(ABC):
    """Abstract Base Class for Image Storage Providers (Local, AWS S3, GCP GCS, HTTP)."""

    @abstractmethod
    def read_image(self, path_or_uri: str) -> np.ndarray:
        """Reads image from storage location and returns BGR/RGB numpy ndarray."""

    @abstractmethod
    def exists(self, path_or_uri: str) -> bool:
        """Checks if file exists at specified path/uri."""
