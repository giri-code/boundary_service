"""Memory-safety regression tests: singleflight, int4 odd shapes."""
import threading

import numpy as np

from app.services.embedding_service import EmbeddingService


def test_singleflight_one_encode_per_key(monkeypatch):
    from app.models.mobile_sam_engine import MobileSAMEngine

    engine = MobileSAMEngine.__new__(MobileSAMEngine)
    from app.models.base import BaseSegmentationEngine
    BaseSegmentationEngine.__init__(engine, name="MobileSAM")
    engine._lock = threading.Lock()
    engine._inflight = {}
    engine._sam_model = None
    engine._load_failed = False
    engine.checkpoint_path = "test-checkpoint"
    engine._load_model = lambda: None  # weights irrelevant: encode/predict mocked

    calls = []
    lock = threading.Lock()

    def fake_encode(image):
        import time
        with lock:
            calls.append(1)
        time.sleep(0.2)
        return {"features": np.zeros((1, 4), dtype=np.float32)}

    monkeypatch.setattr(engine, "encode_image", fake_encode)
    monkeypatch.setattr(
        engine, "predict_from_embedding",
        lambda d, point, level=None: (np.zeros((4, 4), dtype=np.uint8), 0.9),
    )

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    outs = []
    threads = [
        threading.Thread(target=lambda: outs.append(engine.predict_mask(img, (1, 1), cache_key="k", level=0)))
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert len(outs) == 3


def test_int4_odd_shape_roundtrip():
    feats = np.arange(35, dtype=np.float32).reshape(7, 5)
    blob = EmbeddingService.serialize_embedding(
        {"features": feats, "original_size": (7, 5), "input_size": (7, 5)}
    )
    back = EmbeddingService.deserialize_embedding(blob)
    assert back["features"].shape == (7, 5)
