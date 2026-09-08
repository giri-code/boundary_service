import logging
import sys
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


def setup_logger(name: str = "boundary_service", level: str = "INFO") -> logging.Logger:
    """Configures structured application logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request duration and add performance header."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"
        logger.info(
            f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time_ms:.2f}ms"
        )
        return response
