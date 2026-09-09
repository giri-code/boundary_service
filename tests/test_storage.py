from pathlib import Path
import pytest
import numpy as np
from app.storage.factory import StorageProviderFactory
from app.storage.local_provider import LocalStorageProvider
from app.config import settings

def test_local_storage_provider_read():
    provider = LocalStorageProvider()
    sample_path = str(Path(settings.LOCAL_STORAGE_ROOT) / "sample_hat.png")

    assert provider.exists(sample_path) is True
    image = provider.read_image(sample_path)
    assert isinstance(image, np.ndarray)
    assert image.shape[2] == 3

def test_storage_factory_resolution():
    local_p = StorageProviderFactory.get_provider("sample_hat.png")
    assert isinstance(local_p, LocalStorageProvider)

    s3_p = StorageProviderFactory.get_provider("s3://bucket/key.png")
    assert s3_p.__class__.__name__ == "S3StorageProvider"

    http_p = StorageProviderFactory.get_provider("http://example.com/image.png")
    assert http_p.__class__.__name__ == "HTTPStorageProvider"

def test_local_storage_file_not_found():
    provider = LocalStorageProvider()
    with pytest.raises(FileNotFoundError):
        provider.read_image("non_existent_file_12345.png")
