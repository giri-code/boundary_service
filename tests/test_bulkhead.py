"""Bulkhead tests: per-op + overall concurrency limiters for inference/encode.

- Constants pin the intended sizes (10 inference / 3 encode / 14 total).
- Lifespan wiring test boots the real app and asserts the limiters exist
  with the right tokens (guards against an endpoint silently reverting to
  the unbounded default pool).
- Behaviour tests drive _run_in_bulkhead directly with trivial work and
  assert the gates actually cap concurrency (no model/storage needed).
"""
import asyncio
import threading
import time

import anyio

import app.main as main
from app.constants import (
    MAX_CONCURRENT_ENCODE_THREADS,
    MAX_CONCURRENT_INFERENCE_THREADS,
    MAX_CONCURRENT_TOTAL_THREADS,
)


def test_bulkhead_constants():
    assert MAX_CONCURRENT_INFERENCE_THREADS == 10
    assert MAX_CONCURRENT_ENCODE_THREADS == 3
    assert MAX_CONCURRENT_TOTAL_THREADS == 14


def test_lifespan_creates_limiters():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app):
        assert main._inference_limiter.total_tokens == MAX_CONCURRENT_INFERENCE_THREADS
        assert main._encode_limiter.total_tokens == MAX_CONCURRENT_ENCODE_THREADS
        assert main._overall_limiter.total_tokens == MAX_CONCURRENT_TOTAL_THREADS


class _Probe:
    """Sync workload that records peak concurrent executions."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(self, arg):
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        try:
            time.sleep(self.delay)
        finally:
            with self._lock:
                self.current -= 1
        return arg


def _run_concurrent(probe, op_limiter, overall_limiter, n):
    # The helper takes the per-op limiter explicitly but reads the overall
    # gate from the module global (as the endpoints do) — patch just that.
    old_overall = main._overall_limiter
    main._overall_limiter = overall_limiter
    try:
        async def _burst():
            async def _one(i):
                return await main._run_in_bulkhead(probe, i, op_limiter)

            return await asyncio.gather(*[_one(i) for i in range(n)])

        return asyncio.run(_burst())
    finally:
        main._overall_limiter = old_overall


def test_overall_limiter_caps_combined_concurrency():
    probe = _Probe()
    results = _run_concurrent(
        probe,
        op_limiter=anyio.CapacityLimiter(10),  # per-op gate wide open
        overall_limiter=anyio.CapacityLimiter(2),  # overall gate binding
        n=5,
    )
    assert sorted(results) == [0, 1, 2, 3, 4]
    assert probe.peak == 2


def test_per_op_limiter_caps_own_lane():
    probe = _Probe()
    results = _run_concurrent(
        probe,
        op_limiter=anyio.CapacityLimiter(1),  # per-op gate binding
        overall_limiter=anyio.CapacityLimiter(10),  # overall gate wide open
        n=4,
    )
    assert sorted(results) == [0, 1, 2, 3]
    assert probe.peak == 1


def test_missing_limiters_fall_back_to_default_pool():
    old_overall = main._overall_limiter
    main._overall_limiter = None
    try:
        assert asyncio.run(main._run_in_bulkhead(lambda x: x * 2, 21, None)) == 42
    finally:
        main._overall_limiter = old_overall
