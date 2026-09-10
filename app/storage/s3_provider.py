import numpy as np
from .base import BaseStorageProvider
from ..config import settings


class S3StorageProvider(BaseStorageProvider):
    """AWS S3 Object Storage Provider.

    Reads images directly from an S3 bucket using boto3.
    Credentials are resolved via the standard AWS credential chain
    (env vars, ~/.aws/credentials, IAM role, etc.).
    """

    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or settings.S3_BUCKET
        self._s3_client = None

    @property
    def client(self):
        """Lazy-initialise the S3 client."""
        if self._s3_client is None:
            try:
                import boto3

                client_kwargs = {}
                if settings.S3_ENDPOINT:
                    client_kwargs["endpoint_url"] = settings.S3_ENDPOINT
                if settings.S3_REGION:
                    client_kwargs["region_name"] = settings.S3_REGION
                if settings.S3_ACCESS_KEY_ID and settings.S3_SECRET_ACCESS_KEY:
                    client_kwargs["aws_access_key_id"] = settings.S3_ACCESS_KEY_ID
                    client_kwargs["aws_secret_access_key"] = settings.S3_SECRET_ACCESS_KEY

                self._s3_client = boto3.client("s3", **client_kwargs)
            except ImportError:
                raise ImportError(
                    "boto3 is required for the S3 storage provider. "
                    "Install it with: pip install boto3"
                )
        return self._s3_client

    def _parse_s3_uri(self, uri: str):
        """Extract (bucket, key) from an s3:// URI or fall back to configured bucket."""
        if uri.startswith("s3://"):
            parts = uri[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            return bucket, key
        return self.bucket_name, uri

    def read_image(self, path_or_uri: str) -> np.ndarray:
        bucket, key = self._parse_s3_uri(path_or_uri)
        if not bucket or not key:
            raise ValueError(
                f"Invalid S3 URI or missing bucket name: '{path_or_uri}'. "
                "Use 's3://bucket-name/path/to/image.jpg' or set AWS_S3_BUCKET."
            )

        # FIX PERF-3: check Content-Length before reading body to catch oversized files early
        head = self.client.head_object(Bucket=bucket, Key=key)
        content_length_bytes = head.get("ContentLength", 0)
        max_bytes = settings.MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024
        if content_length_bytes > max_bytes:
            raise ValueError(
                f"S3 object size ({content_length_bytes / (1024*1024):.1f} MB) "
                f"exceeds the configured limit ({settings.MAX_IMAGE_FILE_SIZE_MB} MB)."
            )

        response = self.client.get_object(Bucket=bucket, Key=key)
        image_bytes = response["Body"].read()
        from .decode import decode_image_bytes

        image = decode_image_bytes(image_bytes, f"S3: {path_or_uri}")

        # Guard pixel limit
        h, w = image.shape[:2]
        if h * w > settings.MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image dimensions ({w}×{h} = {h * w} px) exceed "
                f"the pixel limit ({settings.MAX_IMAGE_PIXELS} px)."
            )

        return image

    def exists(self, path_or_uri: str) -> bool:
        bucket, key = self._parse_s3_uri(path_or_uri)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
