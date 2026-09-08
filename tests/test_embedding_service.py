import os
import numpy as np
import pytest
from app.services.embedding_service import EmbeddingService, MAGIC_HEADER, FORMAT_VERSION


def test_embedding_serialization_roundtrip(tmp_path):
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


def test_embedding_save_load_delete(tmp_path, monkeypatch):
    photo_id = "test_photo_123"
    test_file_path = str(tmp_path / f"{photo_id}_embed.bin")
    monkeypatch.setattr(EmbeddingService, "get_file_path", classmethod(lambda cls, pid: test_file_path))
    # Disable redis for this local test
    monkeypatch.setattr(EmbeddingService, "get_redis", classmethod(lambda cls: None))
    features = np.ones((1, 256, 64, 64), dtype=np.float32) * 0.5
    sample = {
        "features": features,
        "original_size": (800, 600),
        "input_size": (800, 600),
    }

    EmbeddingService.save(photo_id, sample)
    expected_path = os.path.join(str(tmp_path), f"{photo_id}_embed.bin")
    assert os.path.exists(expected_path)

    loaded = EmbeddingService.load(photo_id)
    assert loaded is not None
    assert np.allclose(loaded["features"], features)
    assert loaded["original_size"] == (800, 600)

    EmbeddingService.delete(photo_id)
    assert not os.path.exists(expected_path)
    assert EmbeddingService.load(photo_id) is None
