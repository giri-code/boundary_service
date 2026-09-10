import pytest
from app.storage.factory import StorageProviderFactory
from app.storage.s3_provider import S3StorageProvider

def test_storage_factory_resolution():
    s3_p1 = StorageProviderFactory.get_provider("sample_hat.png")
    assert isinstance(s3_p1, S3StorageProvider)

    s3_p2 = StorageProviderFactory.get_provider("s3://bucket/key.png")
    assert isinstance(s3_p2, S3StorageProvider)

def test_storage_factory_rejects_http():
    with pytest.raises(ValueError, match="not supported"):
        StorageProviderFactory.get_provider("http://example.com/image.png")
    with pytest.raises(ValueError, match="not supported"):
        StorageProviderFactory.get_provider("https://example.com/image.png")
