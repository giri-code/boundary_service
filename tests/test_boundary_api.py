from pathlib import Path
import pytest
import httpx
from app.main import app
from app.config import settings

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.asyncio
async def test_health_check():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_provider" in data

@pytest.mark.asyncio
async def test_detect_boundary_success():
    sample_path = str(Path(settings.LOCAL_STORAGE_ROOT) / "sample_hat.png")

    payload = {
        "image_path": sample_path,
        "x": 300,
        "y": 200,
        "tolerance": 0.005,
        "model_provider": "opencv"
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload, headers={"X-Internal-Token": settings.INTERNAL_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["outer_boundary"]) >= 3
        assert data["area_pixels"] > 0
        assert "bounding_box" in data
        assert data["image_dimensions"]["width"] == 600
        assert data["image_dimensions"]["height"] == 600

@pytest.mark.asyncio
async def test_detect_boundary_relative_path():
    payload = {
        "image_path": "sample_hat.png",
        "x": 300,
        "y": 200,
        "tolerance": 0.005,
        "model_provider": "opencv"
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload, headers={"X-Internal-Token": settings.INTERNAL_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["outer_boundary"]) >= 3

@pytest.mark.asyncio
async def test_detect_boundary_out_of_bounds():
    sample_path = str(Path(settings.LOCAL_STORAGE_ROOT) / "sample_hat.png")

    payload = {
        "image_path": sample_path,
        "x": 9999,
        "y": 9999
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload, headers={"X-Internal-Token": settings.INTERNAL_API_KEY})
        assert response.status_code == 400
        body = response.json()
        assert "error" in body or "detail" in body

@pytest.mark.asyncio
async def test_detect_boundary_file_not_found():
    payload = {
        "image_path": "/non/existent/image.jpg",
        "x": 10,
        "y": 10
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/boundary", json=payload, headers={"X-Internal-Token": settings.INTERNAL_API_KEY})
        # Absolute paths outside storage root trigger PermissionError (400)
        # or FileNotFoundError (404) depending on whether path escapes root
        assert response.status_code in (400, 404)
 

@pytest.mark.asyncio
async def test_detect_boundary_embedding_not_ready_returns_409():
    payload = {
        "photo_id": "non_existent_photo_9999",
        "x": 10,
        "y": 10,
        "model_provider": "mobile_sam",
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/boundary",
            json=payload,
            headers={"X-Internal-Token": settings.INTERNAL_API_KEY},
        )
        assert response.status_code == 409
        body = response.json()
        assert "Embedding not ready" in (body.get("error", "") or body.get("detail", ""))
