"""Claim-lock tests: exactly-once-effect encoding across racing replicas.

FakeRedis implements the 3-method surface EncodeClaim needs (SET NX EX, GET,
Lua compare-and-del) with a lock, so concurrency behavior is deterministic
with no server and no new test dependencies.
"""
import threading

import pytest
from fastapi import HTTPException

from app.main import _run_encoding
from app.schemas.boundary import EncodeRequest
from app.services.encode_claim import EncodeClaim, CLAIM_TTL_SECONDS
from app.services.embedding_service import EmbeddingService, S3PersistenceError


class FakeRedis:
    """Minimal thread-safe Redis stand-in for claim semantics."""

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def set(self, key, value, nx=False, ex=None):
        with self._lock:
            if nx and key in self._data:
                return None
            self._data[key] = value
            return True

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def eval(self, script, numkeys, key, token, *args):
        # Mirrors both Lua scripts: compare-and-del (release) and
        # compare-and-refresh-TTL (heartbeat).
        with self._lock:
            if self._data.get(key) != token:
                return 0
            if "del" in script:
                del self._data[key]
            else:
                self._data[key] = token  # TTL refresh: ownership reaffirmed
            return 1


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(
        EmbeddingService, "get_redis", classmethod(lambda cls: fake)
    )
    return fake


def _req(photo_id="p1"):
    return EncodeRequest(photo_id=photo_id, image_path="sample_hat.png")


def test_claim_win_loss_release(fake_redis):
    claim = EncodeClaim(fake_redis)
    token = claim.acquire("p1")
    assert token  # won
    assert EncodeClaim(fake_redis).acquire("p1") is None  # lost
    assert claim.release("p1", token) is True
    assert EncodeClaim(fake_redis).acquire("p1")  # free again


def test_stale_token_cannot_release(fake_redis):
    claim = EncodeClaim(fake_redis)
    token = claim.acquire("p1")
    # Simulate TTL rollover: someone else owns the key now.
    fake_redis._data["encode:claim:p1"] = "other-owner"
    assert claim.release("p1", token) is False
    assert fake_redis.get("encode:claim:p1") == "other-owner"


def test_no_redis_proceeds_unclaimed(monkeypatch):
    monkeypatch.setattr(
        EmbeddingService, "get_redis", classmethod(lambda cls: None)
    )
    claim = EncodeClaim(None)
    assert claim.acquire("p1") == ""
    assert claim.release("p1", "") is False


class _CountingEngine:
    name = "counting"
    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def encode_image(self, image):
        import time
        with self._lock:
            self.calls += 1
        time.sleep(0.2)  # widen the race window deterministically
        return {"features": b"x", "original_size": (10, 10), "input_size": (10, 10)}


def test_two_racers_one_compute(monkeypatch, fake_redis):
    engine = _CountingEngine()
    monkeypatch.setattr(
        "app.main.SegmentationModelFactory.get_engine", lambda *a, **k: engine
    )
    monkeypatch.setattr(EmbeddingService, "load", classmethod(lambda cls, pid: None))
    monkeypatch.setattr(EmbeddingService, "save", classmethod(lambda cls, pid, d: None))

    outcomes = []

    def _race():
        try:
            outcomes.append(_run_encoding(_req("dup-photo")))
        except HTTPException as exc:
            outcomes.append(exc.status_code)

    threads = [threading.Thread(target=_race) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert engine.calls == 1
    assert len(outcomes) == 2
    assert all(
        o == 409 or (isinstance(o, dict) and o.get("success") is True) for o in outcomes
    )


def test_loser_performs_zero_image_io(monkeypatch, fake_redis):
    # Pre-hold the claim with a foreign token: our call must 409 before read_image.
    fake_redis.set("encode:claim:held-photo", "foreign-token")
    calls = []
    monkeypatch.setattr(
        "app.storage.factory.StorageProviderFactory.read_image",
        lambda path: calls.append(path) or (_ for _ in ()).throw(AssertionError("must not read")),
    )
    monkeypatch.setattr(EmbeddingService, "load", classmethod(lambda cls, pid: None))
    with pytest.raises(HTTPException) as exc_info:
        _run_encoding(_req("held-photo"))
    assert exc_info.value.status_code == 409
    assert calls == []


def test_s3_failure_raises_instead_of_false_200(monkeypatch, fake_redis):
    engine = _CountingEngine()
    monkeypatch.setattr(
        "app.main.SegmentationModelFactory.get_engine", lambda *a, **k: engine
    )
    monkeypatch.setattr(EmbeddingService, "load", classmethod(lambda cls, pid: None))

    def _boom_save(cls, pid, data):
        raise S3PersistenceError("S3 down in test")

    monkeypatch.setattr(EmbeddingService, "save", classmethod(_boom_save))
    with pytest.raises(HTTPException) as exc_info:
        _run_encoding(_req("s3-fail-photo"))
    assert exc_info.value.status_code == 500
    # Claim released even on failure (no orphaned TTL wait).
    assert fake_redis.get("encode:claim:s3-fail-photo") is None


def test_claim_ttl_sane():
    # Short TTL so dead claims die inside the BullMQ retry window (attempts:3 +
    # 10s-base backoff re-races successfully); live encodes heartbeat-refresh.
    assert CLAIM_TTL_SECONDS == 60


def test_heartbeat_refreshes_owned_claim(fake_redis):
    from app.services.encode_claim import ClaimHeartbeat

    claim = EncodeClaim(fake_redis)
    token = claim.acquire("hb-photo")
    assert claim.heartbeat("hb-photo", token) is True
    # Foreign token cannot extend our claim.
    assert claim.heartbeat("hb-photo", "not-owner") is False
    hb = ClaimHeartbeat(claim, "hb-photo", token)
    hb.__enter__()
    hb.__exit__()
    assert claim.release("hb-photo", token) is True
