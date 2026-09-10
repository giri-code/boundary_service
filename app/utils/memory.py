import gc
import logging
import sys

logger = logging.getLogger("boundary_service.memory")


def get_process_memory_mb() -> float:
    """Returns current process RSS in Megabytes (MB).

    Uses psutil (pinned in requirements.txt) so the gauge goes up AND down
    with actual memory — suitable for autoscale/OOM alerting. Falls back to
    resource.ru_maxrss (peak-only, monotonic) when psutil is unavailable.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        else:
            return usage / 1024
    except Exception:
        return 0.0


def force_garbage_collection():
    """Forces garbage collection and clears CUDA/MPS device cache if available."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
