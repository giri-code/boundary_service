import json
import os
import struct
import numpy as np
import redis
from ..config import settings
from ..utils.logger import logger

MAGIC_HEADER = b"MSAM"
FORMAT_VERSION = 1


class EmbeddingService:
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
        return os.path.join(settings.LOCAL_STORAGE_ROOT, f"{photo_id}_embed.bin")

    @classmethod
    def serialize_embedding(cls, embedding_dict: dict) -> bytes:
        """Serialize embedding dictionary to contiguous binary buffer without pickle."""
        features = embedding_dict["features"]
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)

        meta = {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "original_size": list(embedding_dict.get("original_size", (0, 0))),
            "input_size": list(embedding_dict.get("input_size", (0, 0))),
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        # Format: 4B magic + 1B version + 4B meta_length + meta_bytes + raw tensor bytes
        header = MAGIC_HEADER + struct.pack("!BI", FORMAT_VERSION, len(meta_bytes)) + meta_bytes
        return header + features.tobytes()

    @classmethod
    def deserialize_embedding(cls, data: bytes) -> dict:
        """Zero-copy deserialization using np.frombuffer with validation."""
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
        features = np.frombuffer(data[meta_end:], dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])

        return {
            "features": features,
            "original_size": tuple(meta["original_size"]),
            "input_size": tuple(meta["input_size"]),
        }

    @classmethod
    def save(cls, photo_id: str, embedding_dict: dict):
        data = cls.serialize_embedding(embedding_dict)

        r = cls.get_redis()
        if r is not None:
            try:
                r.setex(f"embedding:{photo_id}", 3600, data)
            except Exception as exc:
                logger.error(f"Failed to save embedding to Redis: {exc}")

        file_path = cls.get_file_path(photo_id)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                f.write(data)
        except Exception as exc:
            logger.error(f"Failed to save embedding to disk: {exc}")

    @classmethod
    def load(cls, photo_id: str) -> dict:
        r = cls.get_redis()
        if r is not None:
            try:
                data = r.get(f"embedding:{photo_id}")
                if data:
                    logger.info(f"Loaded embedding for {photo_id} from Redis")
                    return cls.deserialize_embedding(data)
            except Exception as exc:
                logger.error(f"Failed to load embedding from Redis: {exc}")

        file_path = cls.get_file_path(photo_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                logger.info(f"Loaded embedding for {photo_id} from Disk")
                if r is not None:
                    try:
                        r.setex(f"embedding:{photo_id}", 3600, data)
                    except Exception:
                        pass
                return cls.deserialize_embedding(data)
            except Exception as exc:
                logger.error(f"Failed to load embedding from disk: {exc}")

        return None

    @classmethod
    def delete(cls, photo_id: str):
        r = cls.get_redis()
        if r is not None:
            try:
                r.delete(f"embedding:{photo_id}")
                logger.info(f"Deleted embedding for {photo_id} from Redis")
            except Exception as exc:
                logger.error(f"Failed to delete embedding from Redis: {exc}")

        file_path = cls.get_file_path(photo_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted embedding for {photo_id} from Disk")
            except Exception as exc:
                logger.error(f"Failed to delete embedding from disk: {exc}")
