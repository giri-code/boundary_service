import gc
import logging
import sys

logger = logging.getLogger("boundary_service.memory")

def get_process_memory_mb() -> float:
    """Returns current process memory consumption in Megabytes (MB)."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
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
