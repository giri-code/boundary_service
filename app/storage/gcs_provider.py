import cv2
import numpy as np
from .base import BaseStorageProvider
from ..config import settings
from ..utils.logger import logger


class GCSStorageProvider(BaseStorageProvider):
    """Google Cloud Storage (GCS) Provider.

    Credentials are resolved via Application Default Credentials (ADC):
    GOOGLE_APPLICATION_CREDENTIALS env var, gcloud auth, or attached service account.
    """

    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or settings.GCP_GCS_BUCKET
        self._gcs_client = None

    @property
    def client(self):
        """Lazy-initialise the GCS client."""
        if self._gcs_client is None:
            try:
                from google.cloud import storage
                from google.oauth2 import service_account
                
                if settings.FIREBASE_PRIVATE_KEY and settings.FIREBASE_CLIENT_EMAIL:
                    private_key = settings.FIREBASE_PRIVATE_KEY.replace('\\n', '\n')
                    creds_dict = {
                        "type": "service_account",
                        "project_id": settings.FIREBASE_PROJECT_ID,
                        "private_key": private_key,
                        "client_email": settings.FIREBASE_CLIENT_EMAIL,
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    self._gcs_client = storage.Client(credentials=credentials, project=settings.FIREBASE_PROJECT_ID)
                else:
                    self._gcs_client = storage.Client()
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for the GCS storage provider. "
                    "Install it with: pip install google-cloud-storage"
                )
        return self._gcs_client

    def _parse_gcs_uri(self, uri: str):
        """Extract (bucket, blob_name) from a gs:// URI or fall back to configured bucket."""
        if uri.startswith("gs://"):
            parts = uri[5:].split("/", 1)
            bucket = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            return bucket, blob_name
        return self.bucket_name, uri

    def read_image(self, path_or_uri: str) -> np.ndarray:
        bucket_name, blob_name = self._parse_gcs_uri(path_or_uri)
        if not bucket_name or not blob_name:
            raise ValueError(
                f"Invalid GCS URI or missing bucket name: '{path_or_uri}'. "
                "Use 'gs://bucket-name/path/to/image.jpg' or set GCP_GCS_BUCKET."
            )

        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # FIX PERF-3 (GCS variant): check blob size before downloading
        blob.reload()  # ensures metadata (size) is populated
        max_bytes = settings.MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024
        if blob.size and blob.size > max_bytes:
            raise ValueError(
                f"GCS blob size ({blob.size / (1024*1024):.1f} MB) "
                f"exceeds the configured limit ({settings.MAX_IMAGE_FILE_SIZE_MB} MB)."
            )

        image_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image from GCS: {path_or_uri}")

        # Guard pixel limit
        h, w = image.shape[:2]
        if h * w > settings.MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image dimensions ({w}×{h} = {h * w} px) exceed "
                f"the pixel limit ({settings.MAX_IMAGE_PIXELS} px)."
            )

        return image

    def exists(self, path_or_uri: str) -> bool:
        bucket_name, blob_name = self._parse_gcs_uri(path_or_uri)
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.exists()
        except Exception:
            return False
