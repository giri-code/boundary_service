from pathlib import Path
import pytest
import httpx
from app.main import app
from app.config import settings

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.asyncio
async def test_api_path_traversal_returns_4xx():
    """FUNC-1: API must reject path-traversal attempts with 4xx status."""
    payload = {
        "image_path": "/etc/passwd",
        "x": 10,
        "y": 10,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload, headers={"X-Internal-Token": settings.INTERNAL_API_KEY})
        assert response.status_code in (400, 403, 404)

@pytest.mark.asyncio
async def test_missing_internal_token_rejected():
    """Ensure requests without X-Internal-Token are rejected with 403."""
    payload = {"image_path": "dummy.jpg", "x": 10, "y": 10}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload)
        assert response.status_code == 403
        assert "Missing X-Internal-Token" in response.text

@pytest.mark.asyncio
async def test_invalid_internal_token_rejected():
    """Ensure requests with incorrect X-Internal-Token are rejected with 403."""
    payload = {"image_path": "dummy.jpg", "x": 10, "y": 10}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload, headers={"X-Internal-Token": "wrong-token"})
        assert response.status_code == 403
        assert "Invalid internal token" in response.text

@pytest.mark.asyncio
async def test_encode_missing_internal_token_rejected():
    """POST /encode without X-Internal-Token is rejected with 403."""
    payload = {"photo_id": "photo_123", "image_path": "s3://photos/sample_hat.png"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary/encode", json=payload)
        assert response.status_code == 403
        assert "Missing X-Internal-Token" in response.text

@pytest.mark.asyncio
async def test_encode_invalid_internal_token_rejected():
    """POST /encode with wrong X-Internal-Token is rejected with 403."""
    payload = {"photo_id": "photo_123", "image_path": "s3://photos/sample_hat.png"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary/encode", json=payload, headers={"X-Internal-Token": "wrong-token"})
        assert response.status_code == 403
        assert "Invalid internal token" in response.text

@pytest.mark.asyncio
async def test_delete_embedding_missing_internal_token_rejected():
    """DELETE /embedding/{photo_id} without X-Internal-Token is rejected with 403."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/api/v1/boundary/embedding/photo_123")
        assert response.status_code == 403
        assert "Missing X-Internal-Token" in response.text

@pytest.mark.asyncio
async def test_delete_embedding_invalid_internal_token_rejected():
    """DELETE /embedding/{photo_id} with wrong X-Internal-Token is rejected with 403."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/api/v1/boundary/embedding/photo_123", headers={"X-Internal-Token": "wrong-token"})
        assert response.status_code == 403
        assert "Invalid internal token" in response.text
