"""Per-photo encode claim lock: exactly-once-effect encoding across replicas.

Multiple encoder replicas (or BullMQ retries) racing the same `photo_id` must not
each pay the ~500MB ViT transient. Exactly one winner computes; losers get a
retryable 409 and land on the winner's embedding via the load fast-path.

Mechanism: Redis `SET NX EX` (single shared Redis — the same instance that hosts
embedding tier-1 — is this deployment's coordination point; no Redlock needed).
Owner token + Lua compare-and-del release, so an expired claim can never delete
another owner's fresh claim.

Degradation: Redis unavailable → skip-open (proceed unclaimed, logged). The race
window reopens until Redis recovers; the service never blocks on coordination.
"""
import threading
import time
import uuid

from ..utils.logger import logger

CLAIM_TTL_SECONDS = 60
CLAIM_KEY_PREFIX = "encode:claim:"
HEARTBEAT_INTERVAL_SECONDS = 20

# Atomic release: delete only if we still own the claim (guards TTL rollover
# where another replica acquired the same key after our expiry).
_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)

# Atomic refresh: extend TTL only if we still own the claim.
_REFRESH_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('set', KEYS[1], ARGV[1], 'EX', ARGV[2]) else return 0 end"
)


def claim_key(photo_id: str) -> str:
    return f"{CLAIM_KEY_PREFIX}{photo_id}"


class EncodeClaim:
    """Claim guard with a 3-method Redis surface (test seam: FakeRedis in tests).

    `acquire` returns an owner token on win, `None` on loss (caller → 409), and
    `""` when there is no Redis (proceed unclaimed; `release("")` is a no-op).
    """

    def __init__(self, redis_client):
        self._r = redis_client

    def acquire(self, photo_id: str):
        if self._r is None:
            logger.warning(
                f"Encode claim skipped (no Redis) for photo_id='{photo_id}'; "
                "proceeding unclaimed."
            )
            return ""
        token = uuid.uuid4().hex
        try:
            won = self._r.set(claim_key(photo_id), token, nx=True, ex=CLAIM_TTL_SECONDS)
        except Exception as exc:
            logger.warning(
                f"Encode claim errored for photo_id='{photo_id}' ({exc}); proceeding unclaimed."
            )
            return ""
        if not won:
            logger.info(
                f"Encode claim lost for photo_id='{photo_id}'; another replica is encoding."
            )
            return None
        logger.info(f"Encode claim won for photo_id='{photo_id}'.")
        return token

    def release(self, photo_id: str, token: str) -> bool:
        if not token or self._r is None:
            return False
        try:
            return bool(self._r.eval(_RELEASE_LUA, 1, claim_key(photo_id), token))
        except Exception as exc:
            logger.warning(
                f"Encode claim release failed for photo_id='{photo_id}' ({exc}); "
                "TTL expiry will clean it up."
            )
            return False

    def heartbeat(self, photo_id: str, token: str):
        """Refresh our claim TTL (token-guarded). Returns False if we lost it."""
        if not token or self._r is None:
            return False
        try:
            return bool(
                self._r.eval(
                    _REFRESH_LUA, 1, claim_key(photo_id), token, CLAIM_TTL_SECONDS
                )
            )
        except Exception:
            return False


class ClaimHeartbeat:
    """Keep a won claim alive across long encodes.

    TTL (60s) is deliberately short so a *dead* process's claim dies fast: with
    BullMQ attempts:3 + 10s-base backoff, retries at ~+40s/+90s outlast any dead
    claim and re-race successfully instead of DLQing against it. A *live* encode
    refreshes every 20s, so arbitrarily slow boxes never spuriously lose their claim.
    Daemon thread, token-guarded refresh: stops itself, never extends a lost claim.
    Use as `with ClaimHeartbeat(claim, photo_id, token):` around the compute.
    """

    def __init__(self, claim: EncodeClaim, photo_id: str, token: str):
        self._claim = claim
        self._photo_id = photo_id
        self._token = token
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if not self._token:
            return self

        def _loop():
            while not self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                if not self._claim.heartbeat(self._photo_id, self._token):
                    return

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False
