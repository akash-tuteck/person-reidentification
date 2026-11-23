import secrets
from typing import Optional

from fastapi import HTTPException, Query, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check(candidate: Optional[str]) -> str:
    if not settings.AUTH_ENABLED:
        return ""
    if not candidate or not secrets.compare_digest(candidate, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return candidate


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Header-only check for API endpoints."""
    return _check(api_key)


def verify_api_key_or_query(
    api_key: Optional[str] = Security(api_key_header),
    key: Optional[str] = Query(None),
) -> str:
    """For browser <img>/page URLs, which cannot send headers: also accept ?key=."""
    return _check(api_key or key)
