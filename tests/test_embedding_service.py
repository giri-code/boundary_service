import numpy as np
import pytest
from app.services.embedding_service import EmbeddingService, MAGIC_HEADER, FORMAT_VERSION


def test_embedding_serialization_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.embedding_service.EMBEDDING_PRECISION", "float32")
    features = np.random.randn(1, 256, 64, 64).astype(np.float32)
    embedding_dict = {
        "features": features,
        "original_size": (1080, 1920),
        "input_size": (684, 1024),
    }

    raw_bytes = EmbeddingService.serialize_embedding(embedding_dict)
    assert raw_bytes.startswith(MAGIC_HEADER)
    assert raw_bytes[4] == FORMAT_VERSION

    recovered = EmbeddingService.deserialize_embedding(raw_bytes)
    assert np.allclose(features, recovered["features"])
    assert recovered["original_size"] == (1080, 1920)
    assert recovered["input_size"] == (684, 1024)


def test_embedding_corrupted_validation():
    with pytest.raises(ValueError, match="Invalid embedding buffer"):
        EmbeddingService.deserialize_embedding(b"BAD_HEADER_DATA_12345")

    with pytest.raises(ValueError, match="Corrupted or empty"):
        EmbeddingService.deserialize_embedding(b"")


def _make_fake_s3():
    class FakeS3Storage:
        def __init__(self):
            self.store = {}

        def save_bytes(self, key, data):
            self.store[key] = data

        def read_bytes(self, key):
            return self.store[key]

        def exists(self, key):
            return key in self.store

        def delete(self, key):
            self.store.pop(key, None)

    return FakeS3Storage()


def test_embedding_save_load_delete(tmp_path, monkeypatch):
    photo_id = "test_photo_123"
    # No Redis, no disk tier: S3 fake is the only store.
    monkeypatch.setattr(EmbeddingService, "get_redis", classmethod(lambda cls: None))
    fake_s3 = _make_fake_s3()
    monkeypatch.setattr(EmbeddingService, "get_storage_provider", classmethod(lambda cls: fake_s3))
    features = np.ones((1, 256, 64, 64), dtype=np.float32) * 0.5
    sample = {
        "features": features,
        "original_size": (800, 600),
        "input_size": (800, 600),
    }

    EmbeddingService.save(photo_id, sample)
    assert fake_s3.exists("embeddings/test_photo_123.bin")

    loaded = EmbeddingService.load(photo_id)
    assert loaded is not None
    assert np.allclose(loaded["features"], features)
    assert loaded["original_size"] == (800, 600)

    EmbeddingService.delete(photo_id)
    assert not fake_s3.exists("embeddings/test_photo_123.bin")
    assert EmbeddingService.load(photo_id) is None


def test_embedding_s3_tier_rehydration(tmp_path, monkeypatch):
    fake_s3 = _make_fake_s3()
    monkeypatch.setattr(EmbeddingService, "get_storage_provider", classmethod(lambda cls: fake_s3))
    monkeypatch.setattr(EmbeddingService, "get_redis", classmethod(lambda cls: None))

    features = np.ones((1, 256, 64, 64), dtype=np.float32) * 0.42
    sample = {
        "features": features,
        "original_size": (1200, 800),
        "input_size": (1024, 683),
    }

    # 1. Save embedding - should write to S3
    EmbeddingService.save("s3_photo_999", sample)
    assert fake_s3.exists("embeddings/s3_photo_999.bin")

    # 2. Load embedding - should fetch from S3 (no disk tier involved)
    loaded = EmbeddingService.load("s3_photo_999")
    assert loaded is not None
    assert np.allclose(loaded["features"], features)
    assert loaded["original_size"] == (1200, 800)

    # 3. Delete embedding - should purge from S3
    EmbeddingService.delete("s3_photo_999")
    assert not fake_s3.exists("embeddings/s3_photo_999.bin")
    assert EmbeddingService.load("s3_photo_999") is None


def test_embedding_precision_float16_vs_float32(monkeypatch):
    features = np.random.randn(1, 256, 64, 64).astype(np.float32)
    sample = {
        "features": features,
        "original_size": (1080, 1920),
        "input_size": (684, 1024),
    }

    # Test float32 mode
    monkeypatch.setattr("app.services.embedding_service.EMBEDDING_PRECISION", "float32")
    bytes_f32 = EmbeddingService.serialize_embedding(sample)
    recovered_f32 = EmbeddingService.deserialize_embedding(bytes_f32)
    assert recovered_f32["features"].dtype == np.float32
    assert np.allclose(features, recovered_f32["features"])
    assert len(bytes_f32) > 4_000_000

    # Test float16 mode (should be ~50% the size)
    monkeypatch.setattr("app.services.embedding_service.EMBEDDING_PRECISION", "float16")
    bytes_f16 = EmbeddingService.serialize_embedding(sample)
    recovered_f16 = EmbeddingService.deserialize_embedding(bytes_f16)
    assert recovered_f16["features"].dtype == np.float16
    assert np.allclose(features, recovered_f16["features"].astype(np.float32), rtol=1e-3, atol=1e-3)
    assert len(bytes_f16) < 2_200_000

    # Test int8 mode (should be ~25% the size, < 1.1 MB)
    monkeypatch.setattr("app.services.embedding_service.EMBEDDING_PRECISION", "int8")
    bytes_i8 = EmbeddingService.serialize_embedding(sample)
    recovered_i8 = EmbeddingService.deserialize_embedding(bytes_i8)
    assert recovered_i8["features"].dtype == np.float32
    # Quantization error tolerance: max error is bounded by scale = max/127
    max_val = np.max(np.abs(features))
    assert np.allclose(features, recovered_i8["features"], atol=(max_val / 127.0))
    assert len(bytes_i8) < 1_100_000

    # Test int4 mode (should be ~12.5% the size, ~524 KB)
    monkeypatch.setattr("app.services.embedding_service.EMBEDDING_PRECISION", "int4")
    bytes_i4 = EmbeddingService.serialize_embedding(sample)
    recovered_i4 = EmbeddingService.deserialize_embedding(bytes_i4)
    assert recovered_i4["features"].dtype == np.float32
    # 4-bit error tolerance: bounded by scale = max/7.0
    assert np.allclose(features, recovered_i4["features"], atol=(max_val / 7.0))
    assert len(bytes_i4) < 550_000
