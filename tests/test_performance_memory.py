from pathlib import Path
import time
import pytest
import httpx
from app.main import app
from app.config import settings
from app.utils.memory import get_process_memory_mb, force_garbage_collection

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.asyncio
async def test_memory_stability_repeated_requests():
    sample_path = str(Path(settings.LOCAL_STORAGE_ROOT) / "sample_hat.png")
    payload = {
        "image_path": sample_path,
        "x": 300,
        "y": 200,
        "tolerance": 0.005,
        "model_provider": "opencv"
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Run warm-up loop to allow Python heap/allocator to reach steady state
        for _ in range(10):
            await client.post("/api/v1/boundary", json=payload)
        
        force_garbage_collection()
        initial_memory = get_process_memory_mb()

        # Execute 30 requests in steady state
        for _ in range(30):
            res = await client.post("/api/v1/boundary", json=payload)
            assert res.status_code == 200

        force_garbage_collection()
        final_memory = get_process_memory_mb()

    steady_state_growth = final_memory - initial_memory
    print(f"\nInitial Steady Memory: {initial_memory:.2f} MB | Final Memory: {final_memory:.2f} MB | Delta: {steady_state_growth:.2f} MB")

    # Python's glibc allocator retains freed pages at the process level (RSS-level),
    # so some growth after warm-up is expected and normal in test environments.
    # True leaks would cause unbounded growth proportional to request count.
    assert steady_state_growth < 150.0

@pytest.mark.asyncio
async def test_latency_performance():
    sample_path = str(Path(settings.LOCAL_STORAGE_ROOT) / "sample_hat.png")
    payload = {
        "image_path": sample_path,
        "x": 300,
        "y": 200,
        "tolerance": 0.005,
        "model_provider": "opencv"
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/api/v1/boundary", json=payload)

        start = time.time()
        res = await client.post("/api/v1/boundary", json=payload)
        latency_ms = (time.time() - start) * 1000

        assert res.status_code == 200
        print(f"\nSteady state API Latency: {latency_ms:.2f} ms")
        assert latency_ms < 300.0
