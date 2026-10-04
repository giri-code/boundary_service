import secrets
from fastapi import Request, HTTPException, status
from .config import settings
from .utils.logger import logger


async def verify_internal_token(request: Request):
    """
    Dependency to verify that the incoming request is authorized by the internal backend.
    Checks the 'X-Internal-Token' header against the configured INTERNAL_API_KEY.
    """
    secret = settings.INTERNAL_SERVICE_SECRET or settings.INTERNAL_API_KEY
    if not secret:
        logger.warning(
            "INTERNAL_SERVICE_SECRET is not set in configuration. Rejecting request."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is missing internal security configuration.",
        )

    if secret in settings.REJECTED_SAMPLE_SECRETS:
        # The configured value is a publicly documented sample (committed to
        # the repo and docs). Refuse to honor it — fail closed, never log it.
        logger.warning(
            "INTERNAL_SERVICE_SECRET matches a publicly documented sample value. "
            "Rejecting request: rotate to an operator-supplied secret."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is missing internal security configuration.",
        )

    token = request.headers.get("X-Internal-Token")
    client_host = request.client.host if request.client else "unknown"

    if not token:
        logger.warning(
            f"Internal auth rejected: Missing X-Internal-Token header for {request.method} {request.url.path} from {client_host}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing X-Internal-Token header.",
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(
        token.encode("utf-8"), secret.encode("utf-8")
    ):
        logger.warning(
            f"Internal auth rejected: Invalid X-Internal-Token for {request.method} {request.url.path} from {client_host}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal token.",
        )

    return True

