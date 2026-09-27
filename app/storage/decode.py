"""Shared image-bytes → BGR decoding for all storage providers.

Single place that knows which formats this service can read. OpenCV handles
the common formats; HEIC (iPhone uploads) needs the pillow-heif bridge —
libheif is bundled in the wheel, no system packages required.
"""

import io
import numpy as np
from ..utils.logger import logger


def decode_image_bytes(image_bytes: bytes, source_label: str) -> np.ndarray:
    """Decode raw image bytes to a BGR uint8 ndarray (H, W, 3).

    Raises:
        ValueError: if the bytes cannot be decoded by any available decoder.
    """
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    heic_error = None
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = np.array(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        pass
    except Exception as exc:
        heic_error = str(exc)

    error_msg = f"Failed to decode image from: {source_label}"
    if heic_error:
        logger.warning(f"{error_msg} (HEIC/Pillow fallback error: {heic_error})")
    else:
        logger.warning(error_msg)

    raise ValueError(error_msg)

