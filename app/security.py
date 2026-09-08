import secrets
from fastapi import Request, HTTPException, status
from .config import settings
from .utils.logger import logger


async def verify_internal_token(request: Request):
    """
    Dependency to verify that the incoming request is authorized by the internal backend.
    Checks the 'X-Internal-Token' header against the configured INTERNAL_API_KEY.
    """
    if not settings.INTERNAL_API_KEY:
        logger.warning(
            "INTERNAL_API_KEY is not set in configuration. Rejecting request."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is missing internal security configuration.",
        )

    token = request.headers.get("X-Internal-Token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing X-Internal-Token header.",
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(
        token.encode("utf-8"), settings.INTERNAL_API_KEY.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal token.",
        )

    return True
