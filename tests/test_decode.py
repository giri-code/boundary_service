"""Shared image-decoder tests: cv2 fast path + pillow-heif HEIC bridge."""

from pathlib import Path

import numpy as np
import pytest

from app.storage.decode import decode_image_bytes, probe_image_dimensions

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


def test_probe_returns_header_dimensions_without_full_decode():
    import cv2

    png = np.zeros((16, 12, 3), dtype=np.uint8)
    raw = cv2.imencode(".png", png)[1].tobytes()
    assert probe_image_dimensions(raw) == (12, 16)


def test_probe_garbage_returns_none():
    assert probe_image_dimensions(b"not-an-image-at-all") is None


def test_probe_bomb_raises_before_decode(monkeypatch):
    """A tiny file claiming huge dimensions must fail at probe time."""
    import cv2
    from dataclasses import replace

    import app.config as config_module

    png = np.zeros((16, 12, 3), dtype=np.uint8)
    raw = cv2.imencode(".png", png)[1].tobytes()
    # Shrink the budget below the fixture so the header claim exceeds it.
    # (settings is a frozen dataclass: replace the module binding instead.)
    # Limit 50: PIL raises (192 px > 2x limit); explicit compare covers 1x-2x.
    monkeypatch.setattr(
        config_module, "settings", replace(config_module.settings, MAX_IMAGE_PIXELS=50)
    )
    with pytest.raises(ValueError, match="pixel limit"):
        probe_image_dimensions(raw)


def test_s3_read_image_rejects_bomb_before_decode(monkeypatch):
    """S3 read path must reject on header claim without calling imdecode."""
    import cv2
    from dataclasses import replace

    import app.storage.s3_provider as s3_module
    from app.storage.s3_provider import S3StorageProvider

    png = np.zeros((16, 12, 3), dtype=np.uint8)
    raw = cv2.imencode(".png", png)[1].tobytes()

    provider = S3StorageProvider(bucket_name="test-bucket")

    class FakeBody:
        def read(self):
            return raw

    class FakeClient:
        def head_object(self, Bucket, Key):
            return {"ContentLength": len(raw)}

        def get_object(self, Bucket, Key):
            return {"Body": FakeBody()}

    monkeypatch.setattr(provider, "_s3_client", FakeClient())
    monkeypatch.setattr(
        s3_module, "settings", replace(s3_module.settings, MAX_IMAGE_PIXELS=100)
    )
    # Prove the rejection happens BEFORE full decode: fail loudly if reached.
    import app.storage.decode as decode_module

    def _must_not_decode(*args, **kwargs):
        raise AssertionError("full decode must not run for header-rejected bombs")

    monkeypatch.setattr(decode_module, "decode_image_bytes", _must_not_decode)
    with pytest.raises(ValueError, match="pixel limit"):
        provider.read_image("s3://test-bucket/uploads/p1.png")


def test_heic_pixels_encode():
    """HEIC pixels flow through the real ViT encoder like any other image."""
    from app.models.factory import SegmentationModelFactory

    raw = FIXTURE_HEIC.read_bytes()
    image = decode_image_bytes(raw, "test")
    engine = SegmentationModelFactory.get_engine("mobile_sam")
    embedding = engine.encode_image(image)
    assert embedding["features"].shape == (1, 256, 64, 64)
    assert embedding["original_size"] == (32, 32)
