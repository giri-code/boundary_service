import pytest
from app.storage.factory import StorageProviderFactory
from app.storage.s3_provider import S3StorageProvider
from app.storage.factory import HTTPStorageProvider

def test_storage_factory_resolution():
    s3_p1 = StorageProviderFactory.get_provider("sample_hat.png")
    assert isinstance(s3_p1, S3StorageProvider)

    s3_p2 = StorageProviderFactory.get_provider("s3://bucket/key.png")
    assert isinstance(s3_p2, S3StorageProvider)

    http_p = StorageProviderFactory.get_provider("http://example.com/image.png")
    assert isinstance(http_p, HTTPStorageProvider)
