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


def _request_with_token(token: str):
    from fastapi import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/boundary",
        "headers": [(b"x-internal-token", token.encode("utf-8"))],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_sample_secret_never_authenticates(monkeypatch):
    """A publicly documented sample secret must fail closed even when configured."""
    from dataclasses import replace

    import app.security as security_module
    from app.config import settings as live_settings
    from app.security import verify_internal_token
    from fastapi import HTTPException

    sample = next(iter(live_settings.REJECTED_SAMPLE_SECRETS))
    monkeypatch.setattr(
        security_module,
        "settings",
        replace(
            live_settings,
            INTERNAL_SERVICE_SECRET=sample,
            INTERNAL_API_KEY=sample,
        ),
    )
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_token(_request_with_token(sample))
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_missing_secret_fails_closed(monkeypatch):
    """Empty secret must fail closed with 500, never allow the request through."""
    from dataclasses import replace

    import app.security as security_module
    from app.config import settings as live_settings
    from app.security import verify_internal_token
    from fastapi import HTTPException

    monkeypatch.setattr(
        security_module,
        "settings",
        replace(
            live_settings, INTERNAL_SERVICE_SECRET="", INTERNAL_API_KEY=""
        ),
    )
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_token(_request_with_token("anything"))
    assert exc_info.value.status_code == 500
