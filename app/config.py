import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables.

    All fields are read-only after construction — accidental mutation at runtime
    will raise a FrozenInstanceError, making configuration bugs immediately visible.
    """

    # ── Service Information ────────────────────────────────────────────────────
    APP_NAME: str = field(
        default_factory=lambda: os.getenv(
            "APP_NAME", "Object Boundary Detection Service"
        )
    )
    VERSION: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))
    ENVIRONMENT: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "production")
    )
    DEBUG: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "False").lower() in ("true", "1")
    )
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # ── Security & CORS ────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: tuple = field(
        default_factory=lambda: tuple(os.getenv("ALLOWED_ORIGINS", "*").split(","))
    )
    INTERNAL_SERVICE_SECRET: str = field(
        default_factory=lambda: os.getenv("INTERNAL_SERVICE_SECRET")
        or os.getenv("INTERNAL_API_KEY", "")
    )
    INTERNAL_API_KEY: str = field(
        default_factory=lambda: os.getenv("INTERNAL_SERVICE_SECRET")
        or os.getenv("INTERNAL_API_KEY", "")
    )

    # ── Storage & Upload Settings ──────────────────────────────────────────────
    STORAGE_BACKEND: str = field(
        default_factory=lambda: os.getenv("STORAGE_BACKEND", "auto")
    )
    LOCAL_STORAGE_ROOT: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_STORAGE_ROOT", str(BASE_DIR / "storage")
        )
    )
    MAX_IMAGE_FILE_SIZE_MB: int = field(
        default_factory=lambda: int(os.getenv("MAX_IMAGE_FILE_SIZE_MB", "50"))
    )
    MAX_IMAGE_PIXELS: int = field(
        default_factory=lambda: int(os.getenv("MAX_IMAGE_PIXELS", "36000000"))
    )
    ALLOWED_IMAGE_EXTENSIONS: tuple = field(
        default_factory=lambda: tuple(
            ext.strip().lower()
            for ext in os.getenv(
                "ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp,.heic,.heif,.avif"
            ).split(",")
            if ext.strip()
        )
    )
    MIN_HOLE_AREA_PIXELS: int = field(
        default_factory=lambda: int(os.getenv("MIN_HOLE_AREA_PIXELS", "10"))
    )

    # ── Cloud Credentials (Optional) ──────────────────────────────────────────
    AWS_S3_BUCKET: str = field(default_factory=lambda: os.getenv("AWS_S3_BUCKET", ""))
    GCP_GCS_BUCKET: str = field(default_factory=lambda: os.getenv("GCP_GCS_BUCKET", ""))
    FIREBASE_PROJECT_ID: str = field(
        default_factory=lambda: os.getenv("FIREBASE_PROJECT_ID", "")
    )
    FIREBASE_CLIENT_EMAIL: str = field(
        default_factory=lambda: os.getenv("FIREBASE_CLIENT_EMAIL", "")
    )
    FIREBASE_PRIVATE_KEY: str = field(
        default_factory=lambda: os.getenv("FIREBASE_PRIVATE_KEY", "")
    )

    # ── Segmentation Model Settings ────────────────────────────────────────────
    SEGMENTATION_MODEL_PROVIDER: str = field(
        default_factory=lambda: os.getenv("SEGMENTATION_MODEL_PROVIDER", "mobile_sam")
    )
    MODEL_WEIGHTS_DIR: str = field(
        default_factory=lambda: os.getenv("MODEL_WEIGHTS_DIR", str(BASE_DIR / "models"))
    )
    MOBILE_SAM_CHECKPOINT: str = field(
        default_factory=lambda: os.getenv(
            "MOBILE_SAM_CHECKPOINT", str(BASE_DIR / "models" / "mobile_sam.pt")
        )
    )

    # ── Cache & Performance Optimisation ──────────────────────────────────────
    DEFAULT_POLYGON_TOLERANCE: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_POLYGON_TOLERANCE", "0.005"))
    )
    ENABLE_EMBEDDING_CACHE: bool = field(
        default_factory=lambda: os.getenv("ENABLE_EMBEDDING_CACHE", "True").lower()
        in ("true", "1")
    )
    EMBEDDING_CACHE_SIZE: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_CACHE_SIZE", "15"))
    )
    ENABLE_ROI_CROPPING: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ROI_CROPPING", "True").lower()
        in ("true", "1")
    )
    MAX_ROI_DIMENSION: int = field(
        default_factory=lambda: int(os.getenv("MAX_ROI_DIMENSION", "1024"))
    )
    # TTL in seconds for caching health-check memory reading (avoids per-probe syscall)
    HEALTH_CACHE_TTL_SECONDS: int = field(
        default_factory=lambda: int(os.getenv("HEALTH_CACHE_TTL_SECONDS", "5"))
    )

    # ── External Services ──────────────────────────────────────────────────────
    REDIS_URL: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    )


settings = Settings()
