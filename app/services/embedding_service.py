import json
import struct
import threading
import numpy as np
import redis
from ..config import settings
from ..constants import EMBEDDING_PRECISION
from ..utils.logger import logger


class S3PersistenceError(RuntimeError):
    """Durable-truth write failed: embedding exists in Redis at best.

    Callers MUST NOT report success when this is raised — a Redis-only embedding
    vanishes at TTL expiry. Fail loud (HTTP 500 → job retry → DLQ) instead.
    """

MAGIC_HEADER = b"MSAM"
FORMAT_VERSION = 1

# Redis TTL for the embedding hot tier: 15 minutes. The working set is the
# photos being actively viewed right now — anything idle longer falls out of
# Redis (LRU) and rehydrates from S3 on next click (409 → self-heal re-encode
# as the last resort). Short TTL keeps the Redis footprint small so the
# queue (TTL-less, unevictable) never fights the cache for memory.
EMBEDDING_TTL_SECONDS = 900


class EmbeddingService:
    """Photo-embedding cache (Redis hot tier + S3 durable tier).

    Invariant: photo_ids are immutable — one photo_id always maps to the same
    image bytes. Cached embeddings therefore never go stale and need no
    invalidation; the Redis TTL (EMBEDDING_TTL_SECONDS) is a refresh window only. No container-local
    disk tier by design (ephemeral across deploys, ENOSPC risk, third copy to keep
    coherent) — Redis serves hot reads, S3 is the durable truth.
    """

    _redis_client = None
    _redis_lock = threading.Lock()

    @classmethod
    def get_redis(cls):
        if cls._redis_client is None:
            with cls._redis_lock:
                if cls._redis_client is None:
                    try:
                        # rediss:// (managed Redis with TLS) is negotiated by
                        # from_url automatically; keep verification on unless
                        # REDIS_TLS_REJECT_UNAUTHORIZED=0 (staging self-signed
                        # only — never in prod). Plain redis:// is unaffected.
                        _url = (settings.REDIS_URL or "").strip()
                        _extra: dict = {
                            "socket_connect_timeout": 2,
                            "socket_timeout": 5,
                            "retry_on_timeout": True,
                            # Recycle dead TCP without hanging request paths.
                            "socket_keepalive": True,
                            "health_check_interval": 30,
                        }
                        if _url.lower().startswith("rediss://"):
                            import os

                            _insecure = os.getenv("REDIS_TLS_REJECT_UNAUTHORIZED", "")
                            if _insecure == "0" or _insecure.lower() == "false":
                                _extra["ssl_cert_reqs"] = "none"
                        cls._redis_client = redis.from_url(_url, **_extra)
                    except Exception as exc:
                        # Never log REDIS_URL verbatim: it may embed credentials
                        # (redis://:password@host). Log endpoint host only (L10).
                        try:
                            from urllib.parse import urlsplit

                            _host = urlsplit(settings.REDIS_URL).hostname or "redis"
                        except Exception:
                            _host = "redis"
                        logger.warning(f"Failed to connect to Redis at {_host}: {exc}")
        return cls._redis_client

    @classmethod
    def serialize_embedding(cls, embedding_dict: dict) -> bytes:
        """Serialize embedding dictionary to contiguous binary buffer without pickle."""
        features = embedding_dict["features"]
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        meta = {
            "shape": list(features.shape),
            "original_size": list(embedding_dict.get("original_size", (0, 0))),
            "input_size": list(embedding_dict.get("input_size", (0, 0))),
        }

        if EMBEDDING_PRECISION == "int4":
            max_val = float(np.max(np.abs(features))) or 1.0
            scale = max_val / 7.0
            q = np.clip(np.round(features / scale), -8, 7).astype(np.int8)
            flat = q.ravel()
            # Odd element counts would break the pairwise nibble pack below
            # (broadcast error) — pad one zero nibble and record the true length.
            meta["flat_len"] = int(flat.size)
            if flat.size % 2:
                flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
            u4 = (flat + 8).astype(np.uint8)
            packed = (u4[0::2] & 0x0F) | ((u4[1::2] & 0x0F) << 4)
            raw_payload = packed.tobytes()
            meta["scale"] = scale
            meta["dtype"] = "int4"
        elif EMBEDDING_PRECISION == "int8":
            max_val = float(np.max(np.abs(features))) or 1.0
            scale = max_val / 127.0
            features = np.clip(np.round(features / scale), -128, 127).astype(np.int8)
            raw_payload = features.tobytes()
            meta["scale"] = scale
            meta["dtype"] = "int8"
        elif EMBEDDING_PRECISION == "float16":
            features = features.astype(np.float16)
            raw_payload = features.tobytes()
            meta["dtype"] = "float16"
        else:
            features = features.astype(np.float32)
            raw_payload = features.tobytes()
            meta["dtype"] = "float32"

        meta_bytes = json.dumps(meta).encode("utf-8")
        # Format: 4B magic + 1B version + 4B meta_length + meta_bytes + raw tensor bytes
        header = MAGIC_HEADER + struct.pack("!BI", FORMAT_VERSION, len(meta_bytes)) + meta_bytes
        return header + raw_payload

    @classmethod
    def deserialize_embedding(cls, data: bytes) -> dict:
        """Deserialization using np.frombuffer with .copy() for writable tensor compatibility."""
        if not data or len(data) < 9:
            raise ValueError("Corrupted or empty embedding binary buffer")

        if not data.startswith(MAGIC_HEADER):
            raise ValueError("Invalid embedding buffer: missing MSAM magic header")

        version, meta_len = struct.unpack("!BI", data[4:9])
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported embedding version: {version}")

        meta_end = 9 + meta_len
        if len(data) < meta_end:
            raise ValueError("Truncated embedding metadata buffer")

        meta = json.loads(data[9:meta_end].decode("utf-8"))
        dtype_str = meta.get("dtype", "float32")

        if dtype_str == "int4":
            # Fused dequant: one preallocated fp32 array, ufunc out= writes and
            # in-place scaling. Peak ≈ 4.2MB + one 0.5MB nibble temp instead of
            # ~15MB across low/high/unpacked/fp32/mul/copy temporaries.
            # Math order ((q - 8) * scale) matches the legacy path bit-exactly.
            packed = np.frombuffer(data[meta_end:], dtype=np.uint8)
            scale = meta["scale"]
            flat = np.empty(len(packed) * 2, dtype=np.float32)
            flat[0::2] = (packed & 0x0F).astype(np.float32)
            flat[0::2] -= 8
            flat[0::2] *= scale
            flat[1::2] = ((packed >> 4) & 0x0F).astype(np.float32)
            flat[1::2] -= 8
            flat[1::2] *= scale
            # Truncate padding nibble if the original element count was odd.
            flat_len = meta.get("flat_len")
            if flat_len is not None:
                flat = flat[:flat_len]
            features = flat.reshape(meta["shape"])
        elif dtype_str == "int8":
            raw = np.frombuffer(data[meta_end:], dtype=np.int8)
            features = raw.astype(np.float32).reshape(meta["shape"])
            features *= meta["scale"]
        else:
            features = np.frombuffer(data[meta_end:], dtype=np.dtype(dtype_str)).reshape(meta["shape"]).copy()
            if meta.get("scale") is not None:
                features = (features.astype(np.float32) * meta["scale"]).astype(np.float32)

        return {
            "features": features,
            "original_size": tuple(meta["original_size"]),
            "input_size": tuple(meta["input_size"]),
        }

    @classmethod
    def _is_missing_key_error(cls, exc: Exception) -> bool:
        resp = getattr(exc, "response", None) or {}
        code = ((resp.get("Error", None) or {}).get("Code", "") or "")
        return (
            isinstance(exc, FileNotFoundError)
            or code in ("NoSuchKey", "404", "NotFound", "NoSuchBucket")
        )

    @classmethod
    def _drop_corrupt_copies(cls, photo_id: str, drop_s3: bool) -> None:
        """Delete poisoned copies so the next encode heals instead of 409-looping."""
        r = cls.get_redis()
        if r is not None:
            try:
                r.delete(f"embedding:{photo_id}")
            except Exception:
                pass
        if drop_s3:
            try:
                cls.delete(photo_id)
            except Exception:
                pass

    @classmethod
    def _deserialize_or_drop(cls, data: bytes, photo_id: str, drop_s3: bool):
        """Deserialize, or on corruption drop that tier's copy and report a miss.

        Only the corrupt tier is dropped (a poisoned Redis entry must not destroy
        a healthy S3 copy); S3 corruption drops everything since nothing sits below.
        Returns None on miss/corruption so callers fall through to the next tier.
        """
        try:
            return cls.deserialize_embedding(data)
        except ValueError as exc:
            logger.warning(
                f"Dropping corrupted embedding copy for photo_id='{photo_id}' "
                f"(drop_s3={drop_s3}): {exc}"
            )
            cls._drop_corrupt_copies(photo_id, drop_s3=drop_s3)
            return None

    @classmethod
    def get_s3_key(cls, photo_id: str) -> str:
        return f"embeddings/{photo_id}.bin"

    @classmethod
    def get_storage_provider(cls):
        from ..storage.factory import StorageProviderFactory
        try:
            return StorageProviderFactory.get_provider()
        except Exception as exc:
            logger.warning(f"Could not get storage provider for embeddings: {exc}")
            return None

    @classmethod
    def save(cls, photo_id: str, embedding_dict: dict):
        data = cls.serialize_embedding(embedding_dict)

        # 1. Hot cache: Redis (fast sub-10ms lookup, 15-min working set)
        r = cls.get_redis()
        if r is not None:
            try:
                r.setex(f"embedding:{photo_id}", EMBEDDING_TTL_SECONDS, data)
            except Exception as exc:
                logger.error(
                    f"Failed to save embedding to Redis for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

        # 2. Permanent durable storage: Cloudflare R2 / S3.
        # Fail-LOUD tier (unlike Redis above): a Redis-only success vanishes at
        # TTL expiry, so _run_encoding must see this as an error, never a 200.
        provider = cls.get_storage_provider()
        if provider and hasattr(provider, "save_bytes"):
            s3_key = cls.get_s3_key(photo_id)
            try:
                provider.save_bytes(s3_key, data)
                logger.info(f"Persisted embedding to S3 at '{s3_key}' for photo_id='{photo_id}'")
            except Exception as exc:
                logger.error(
                    f"Failed to save embedding to S3 at '{s3_key}' for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )
                raise S3PersistenceError(
                    f"Failed to persist embedding to S3 at '{s3_key}': {exc}"
                ) from exc

    @classmethod
    def load(cls, photo_id: str) -> dict:
        # Tier 1: Redis cache (sub-10ms)
        r = cls.get_redis()
        if r is not None:
            try:
                data = r.get(f"embedding:{photo_id}")
                if data:
                    logger.debug(f"Loaded embedding for {photo_id} from Redis")
                    hit = cls._deserialize_or_drop(data, photo_id, drop_s3=False)
                    if hit is not None:
                        # Probabilistic TTL refresh: hot photos must not synchronously
                        # rehydrate every 15 min (click stampede under limiter=1), but a
                        # write per click is wasteful — refresh ~10% of hits. EXPIRE,
                        # not SETEX: re-sending multi-MB payloads just to bump a TTL
                        # wastes bandwidth; the value is already present.
                        try:
                            import random

                            if random.random() < 0.10:
                                r.expire(f"embedding:{photo_id}", EMBEDDING_TTL_SECONDS)
                        except Exception:
                            pass
                        return hit
                    # Corrupt Redis copy dropped above; fall through to S3.
            except Exception as exc:
                logger.error(
                    f"Failed to load embedding from Redis for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

        # Tier 2: Permanent R2 / S3 storage (re-hydrate Redis on hit).
        # Single round trip: try GET directly instead of exists()+GET (also kills
        # the delete-in-between TOCTOU — a miss is a miss either way).
        provider = cls.get_storage_provider()
        if provider and hasattr(provider, "read_bytes"):
            s3_key = cls.get_s3_key(photo_id)
            try:
                data = provider.read_bytes(s3_key)
                logger.info(f"Loaded embedding for {photo_id} from S3 ('{s3_key}')")
                hit = cls._deserialize_or_drop(data, photo_id, drop_s3=True)
                if hit is None:
                    return None
                # Re-populate Redis cache for future fast clicks
                if r is not None:
                    try:
                        r.setex(f"embedding:{photo_id}", EMBEDDING_TTL_SECONDS, data)
                    except Exception:
                        pass
                return hit
            except Exception as exc:
                if cls._is_missing_key_error(exc):
                    logger.debug(f"Embedding miss on S3 at '{s3_key}' for photo_id='{photo_id}'")
                    return None
                logger.error(
                    f"Failed to load embedding from S3 at '{s3_key}' for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

        return None

    @classmethod
    def delete(cls, photo_id: str) -> bool:
        # L2: return False on confirmed durable-delete failure so the API
        # layer can 500 (controller retries). Legacy fakes returning None
        # count as success — only an explicit False fails.
        ok = True
        # 1. Delete from Redis
        r = cls.get_redis()
        if r is not None:
            try:
                r.delete(f"embedding:{photo_id}")
                logger.debug(f"Deleted embedding for {photo_id} from Redis")
            except Exception as exc:
                ok = False
                logger.error(
                    f"Failed to delete embedding from Redis for photo_id='{photo_id}'",
                    exc_info=True,
                )

        # 2. Delete from R2 / S3
        provider = cls.get_storage_provider()
        if provider and hasattr(provider, "delete"):
            s3_key = cls.get_s3_key(photo_id)
            try:
                result = provider.delete(s3_key)
                if result is False:
                    ok = False
                    logger.error(f"S3 embedding delete returned failure for '{s3_key}'")
                else:
                    logger.info(f"Deleted embedding for {photo_id} from S3 ('{s3_key}')")
            except Exception as exc:
                ok = False
                logger.error(
                    f"Failed to delete embedding from S3 at '{s3_key}' for photo_id='{photo_id}'",
                    exc_info=True,
                )
        return ok

