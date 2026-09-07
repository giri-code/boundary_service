from pathlib import Path
import pytest
import httpx
from app.main import app
from app.config import settings
from app.storage.local_provider import LocalStorageProvider

@pytest.fixture
def anyio_backend():
    return 'asyncio'

def test_path_traversal_blocked():
    """FUNC-1: paths that escape LOCAL_STORAGE_ROOT must be rejected."""
    provider = LocalStorageProvider()
    # Classic path-traversal attempt
    with pytest.raises(PermissionError):
        provider.read_image("../../../etc/passwd")

def test_path_traversal_absolute_blocked():
    """FUNC-1: absolute paths outside storage root must be rejected."""
    provider = LocalStorageProvider()
    with pytest.raises(PermissionError):
        provider.read_image("/etc/passwd")

def test_path_inside_storage_root_allowed():
    """FUNC-1: valid paths inside storage root must resolve correctly."""
    provider = LocalStorageProvider()
    sample_path = str(Path(settings.LOCAL_STORAGE_ROOT) / "sample_hat.png")
    assert provider.exists(sample_path) is True

@pytest.mark.asyncio
async def test_api_path_traversal_returns_4xx():
    """FUNC-1: API must reject path-traversal attempts with 4xx status."""
    payload = {
        "image_path": "/etc/passwd",
        "x": 10,
        "y": 10,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload)
        assert response.status_code in (400, 403, 404)
