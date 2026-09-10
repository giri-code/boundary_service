import cv2
from pathlib import Path
import pytest
from unittest.mock import patch
from botocore.exceptions import ClientError

from app.storage.factory import StorageProviderFactory

SAMPLE_HAT_PATH = Path(__file__).parent.parent / "storage" / "sample_hat.png"

# Captured before any mocking: the genuine factory entrypoint.
_REAL_READ_IMAGE = StorageProviderFactory.read_image

@pytest.fixture(autouse=True)
def mock_storage_read_image():
    """Auto-mock StorageProviderFactory.read_image for unit tests."""
    sample_img = cv2.imread(str(SAMPLE_HAT_PATH))

    def fake_read_image(path_or_uri: str):
        if "passwd" in path_or_uri or "traversal" in path_or_uri:
            raise PermissionError(f"Access denied: '{path_or_uri}' contains invalid directory traversal.")
        if "non_existent" in path_or_uri or "non/existent" in path_or_uri:
            raise FileNotFoundError(f"Image not found: {path_or_uri}")
        return sample_img

    with patch("app.storage.factory.StorageProviderFactory.read_image", side_effect=fake_read_image):
        yield


class _FakeS3Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """In-memory boto3 S3 stand-in: exercises the real provider logic with no network."""

    def __init__(self):
        self.blobs: dict = {}
        self.length_overrides: dict = {}
        self.head_calls: list = []
        self.get_calls: list = []

    def _missing(self, op: str):
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            op,
        )

    def head_object(self, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        if (Bucket, Key) in self.length_overrides:
            return {"ContentLength": self.length_overrides[(Bucket, Key)]}
        if (Bucket, Key) not in self.blobs:
            self._missing("HeadObject")
        return {"ContentLength": len(self.blobs[(Bucket, Key)])}

    def get_object(self, Bucket, Key):
        self.get_calls.append((Bucket, Key))
        if (Bucket, Key) not in self.blobs:
            self._missing("GetObject")
        return {"Body": _FakeS3Body(self.blobs[(Bucket, Key)])}


@pytest.fixture
def real_s3_stack():
    """Opt out of the autouse mock: real factory + real S3 provider, fake boto3 transport.

    Yields (provider, fake_client). Restores the factory singleton cache afterwards
    so other tests are unaffected.
    """
    from app.storage.s3_provider import S3StorageProvider

    saved_instances = dict(StorageProviderFactory._instances)
    provider = S3StorageProvider(bucket_name="test-bucket")
    fake = FakeS3Client()
    provider._s3_client = fake
    StorageProviderFactory._instances["s3"] = provider
    with patch(
        "app.storage.factory.StorageProviderFactory.read_image", _REAL_READ_IMAGE
    ):
        yield provider, fake
    StorageProviderFactory._instances.clear()
    StorageProviderFactory._instances.update(saved_instances)
