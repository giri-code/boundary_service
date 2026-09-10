"""Shared image-decoder tests: cv2 fast path + pillow-heif HEIC bridge."""

from pathlib import Path

import numpy as np
import pytest

from app.storage.decode import decode_image_bytes

FIXTURE_HEIC = Path(__file__).parent / "fixtures" / "sample.heic"


def test_decode_png_cv2_path(tmp_path):
    import cv2

    png = np.zeros((16, 12, 3), dtype=np.uint8)
    png[4:12, 3:9] = (0, 0, 255)
    raw = cv2.imencode(".png", png)[1].tobytes()
    image = decode_image_bytes(raw, "test")
    assert image.shape == (16, 12, 3)
    assert image.dtype == np.uint8


def test_decode_heic_bridge():
    raw = FIXTURE_HEIC.read_bytes()
    # Sanity: stock OpenCV cannot read this file — the bridge is load-bearing.
    import cv2

    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is None
    image = decode_image_bytes(raw, "test")
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8


def test_decode_garbage_rejected():
    with pytest.raises(ValueError, match="Failed to decode"):
        decode_image_bytes(b"not-an-image-at-all", "test")


def test_heic_pixels_encode():
    """HEIC pixels flow through the real ViT encoder like any other image."""
    from app.models.factory import SegmentationModelFactory

    raw = FIXTURE_HEIC.read_bytes()
    image = decode_image_bytes(raw, "test")
    engine = SegmentationModelFactory.get_engine("mobile_sam")
    embedding = engine.encode_image(image)
    assert embedding["features"].shape == (1, 256, 64, 64)
    assert embedding["original_size"] == (32, 32)
