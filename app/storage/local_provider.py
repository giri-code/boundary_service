import os
from pathlib import Path
import cv2
import numpy as np
from .base import BaseStorageProvider
from ..config import settings


class LocalStorageProvider(BaseStorageProvider):
    """Local File System Storage Provider.

    Security: all resolved paths are validated to lie within LOCAL_STORAGE_ROOT,
    preventing path-traversal attacks (FUNC-1).
    """

    def __init__(self, root_dir: str = None):
        self.root_dir = Path(root_dir or settings.LOCAL_STORAGE_ROOT).resolve()
        os.makedirs(self.root_dir, exist_ok=True)

    def _resolve_path(self, path_or_uri: str) -> Path:
        clean_path = path_or_uri.replace("file://", "")
        p = Path(clean_path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.root_dir / clean_path).resolve()

        # FIX FUNC-1: reject any path that escapes the storage root
        try:
            resolved.relative_to(self.root_dir)
        except ValueError:
            raise PermissionError(
                f"Access denied: '{path_or_uri}' resolves outside the "
                f"allowed storage root '{self.root_dir}'."
            )

        return resolved

    def read_image(self, path_or_uri: str) -> np.ndarray:
        resolved = self._resolve_path(path_or_uri)
        if not resolved.exists():
            raise FileNotFoundError(f"Image not found on local filesystem: {resolved}")

        # Guard: file size check before decoding
        file_size_mb = resolved.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.MAX_IMAGE_FILE_SIZE_MB:
            raise ValueError(
                f"Image file size ({file_size_mb:.1f} MB) exceeds the "
                f"configured limit ({settings.MAX_IMAGE_FILE_SIZE_MB} MB)."
            )

        image = cv2.imread(str(resolved))
        if image is None:
            raise ValueError(f"Failed to decode image from: {resolved}")

        # Guard: pixel count check after decoding
        h, w = image.shape[:2]
        if h * w > settings.MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image dimensions ({w}×{h} = {h * w} px) exceed the "
                f"configured pixel limit ({settings.MAX_IMAGE_PIXELS} px)."
            )

        return image

    def exists(self, path_or_uri: str) -> bool:
        try:
            return self._resolve_path(path_or_uri).exists()
        except (PermissionError, Exception):
            return False
