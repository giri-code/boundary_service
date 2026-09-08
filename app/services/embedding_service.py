import os
import pickle
import redis
from ..config import settings
from ..utils.logger import logger


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
        return os.path.join(settings.LOCAL_STORAGE_ROOT, f"{photo_id}_embed.pkl")

    @classmethod
    def save(cls, photo_id: str, embedding_dict: dict):
        data = pickle.dumps(embedding_dict)

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
                    return pickle.loads(data)
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
                return pickle.loads(data)
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
