import json
import os
import struct
import numpy as np
import redis
from ..config import settings
from ..constants import EMBEDDING_PRECISION
from ..utils.logger import logger

MAGIC_HEADER = b"MSAM"
FORMAT_VERSION = 1


class EmbeddingService:
    """Photo-embedding cache (Redis + disk).

    Invariant: photo_ids are immutable — one photo_id always maps to the same
    image bytes. Cached embeddings therefore never go stale and need no
    invalidation; the Redis 3600s TTL is a refresh window only.
    """

    _redis_client = None

    @classmethod
    def get_redis(cls):
        if cls._redis_client is None:
            try:
                cls._redis_client = redis.from_url(settings.REDIS_URL)
            except Exception as exc:
                logger.warning(
                    f"Failed to connect to Redis at {settings.REDIS_URL}: {exc}"
                )
        return cls._redis_client

    @classmethod
    def get_file_path(cls, photo_id: str) -> str:
        from ..config import BASE_DIR
        cache_dir = BASE_DIR / "cache"
        os.makedirs(cache_dir, exist_ok=True)
        return str(cache_dir / f"{photo_id}_embed.bin")

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

        # 1. Hot cache: Redis (fast sub-10ms lookup, TTL 24 hours = 86400s)
        r = cls.get_redis()
        if r is not None:
            try:
                r.setex(f"embedding:{photo_id}", 86400, data)
            except Exception as exc:
                logger.error(
                    f"Failed to save embedding to Redis for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

        # 2. Local disk cache (quick fallback within container)
        file_path = cls.get_file_path(photo_id)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                f.write(data)
        except Exception as exc:
            logger.error(
                f"Failed to save embedding to disk for photo_id='{photo_id}' at '{file_path}': {exc}",
                exc_info=True,
            )

        # 3. Permanent durable storage: Cloudflare R2 / S3
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

    @classmethod
    def load(cls, photo_id: str) -> dict:
        # Tier 1: Redis cache (sub-10ms)
        r = cls.get_redis()
        if r is not None:
            try:
                data = r.get(f"embedding:{photo_id}")
                if data:
                    logger.debug(f"Loaded embedding for {photo_id} from Redis")
                    return cls.deserialize_embedding(data)
            except Exception as exc:
                logger.error(
                    f"Failed to load embedding from Redis for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

        # Tier 2: Local container disk
        file_path = cls.get_file_path(photo_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                logger.debug(f"Loaded embedding for {photo_id} from Disk")
                if r is not None:
                    try:
                        r.setex(f"embedding:{photo_id}", 86400, data)
                    except Exception:
                        pass
                return cls.deserialize_embedding(data)
            except Exception as exc:
                logger.error(
                    f"Failed to load embedding from disk for photo_id='{photo_id}' at '{file_path}': {exc}",
                    exc_info=True,
                )

        # Tier 3: Permanent R2 / S3 storage (re-hydrate Redis and local disk)
        provider = cls.get_storage_provider()
        if provider and hasattr(provider, "read_bytes"):
            s3_key = cls.get_s3_key(photo_id)
            try:
                if provider.exists(s3_key):
                    data = provider.read_bytes(s3_key)
                    logger.info(f"Loaded embedding for {photo_id} from S3 ('{s3_key}')")
                    # Re-populate Redis cache for future fast clicks
                    if r is not None:
                        try:
                            r.setex(f"embedding:{photo_id}", 86400, data)
                        except Exception:
                            pass
                    # Re-populate local disk
                    try:
                        with open(file_path, "wb") as f:
                            f.write(data)
                    except Exception:
                        pass
                    return cls.deserialize_embedding(data)
            except Exception as exc:
                logger.error(
                    f"Failed to load embedding from S3 at '{s3_key}' for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

        return None

    @classmethod
    def delete(cls, photo_id: str):
        # 1. Delete from Redis
        r = cls.get_redis()
        if r is not None:
            try:
                r.delete(f"embedding:{photo_id}")
                logger.debug(f"Deleted embedding for {photo_id} from Redis")
            except Exception as exc:
                logger.error(
                    f"Failed to delete embedding from Redis for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

        # 2. Delete from Disk
        file_path = cls.get_file_path(photo_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Deleted embedding for {photo_id} from Disk")
            except Exception as exc:
                logger.error(
                    f"Failed to delete embedding from disk for photo_id='{photo_id}' at '{file_path}': {exc}",
                    exc_info=True,
                )

        # 3. Delete from R2 / S3
        provider = cls.get_storage_provider()
        if provider and hasattr(provider, "delete"):
            s3_key = cls.get_s3_key(photo_id)
            try:
                provider.delete(s3_key)
                logger.info(f"Deleted embedding for {photo_id} from S3 ('{s3_key}')")
            except Exception as exc:
                logger.error(
                    f"Failed to delete embedding from S3 at '{s3_key}' for photo_id='{photo_id}': {exc}",
                    exc_info=True,
                )

