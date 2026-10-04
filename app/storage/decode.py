"""Shared image-bytes → BGR decoding for all storage providers.

Single place that knows which formats this service can read. OpenCV handles
the common formats; HEIC (iPhone uploads) needs the pillow-heif bridge —
libheif is bundled in the wheel, no system packages required.
"""

import io
import numpy as np
from ..utils.logger import logger


def probe_image_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Read (width, height) from the image header WITHOUT full decode.

    PIL's Image.open parses only headers, so a highly-compressible bomb is
    rejected on its dimension claim before any H*W*3 buffer is allocated.
    Returns None when headers are unparseable (caller falls through to the
    normal decode path, which raises ValueError on its own).
    """
    try:
        from PIL import Image
        from PIL.Image import DecompressionBombError
    except ImportError:
        return None
    try:
        # Enforce our own pixel budget at probe time (PIL's default is ~178M).
        # Restored afterwards: this module must not change global PIL state.
        from ..config import settings

        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
    except Exception:
        previous_limit = None
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
    except ImportError:
        pass
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Header-only parse: img.size is available without img.load().
            return int(img.size[0]), int(img.size[1])
    except DecompressionBombError as exc:
        # The header alone proves the image exceeds budget — fail closed
        # instead of falling through to a full decode.
        from ..config import settings

        raise ValueError(
            f"Image exceeds the pixel limit ({settings.MAX_IMAGE_PIXELS} px)."
        ) from exc
    except Exception:
        return None
    finally:
        if previous_limit is not None:
            Image.MAX_IMAGE_PIXELS = previous_limit


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

