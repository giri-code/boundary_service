import logging
import os
import sys
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Endpoints that are probed frequently by load balancers and health monitors
_NOISE_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}


def setup_logger(name: str = "boundary_service", level: str = None) -> logging.Logger:
    """Configures structured application logger."""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
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
    """Middleware to log request duration and add performance header.
    
    Filters out noisy high-frequency health probes (unless they error) to keep
    production logs clean and focused on actual service traffic and errors.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"

        path = request.url.path
        status_code = response.status_code

        # Suppress routine 2xx/3xx access logs on health check endpoints
        if path in _NOISE_PATHS and status_code < 400 and not logger.isEnabledFor(logging.DEBUG):
            return response

        log_msg = f"{request.method} {path} - Status: {status_code} - Time: {process_time_ms:.2f}ms"
        if status_code >= 500:
            logger.error(log_msg)
        elif status_code >= 400:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return response

