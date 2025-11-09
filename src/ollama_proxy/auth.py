from __future__ import annotations

from fastapi import HTTPException


def _extract_bearer(authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1]
    return None


def require_api_key(expected: str, authorization: str | None) -> None:
    token = _extract_bearer(authorization)
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def optional_api_key(expected: str, authorization: str | None) -> None:
    token = _extract_bearer(authorization)
    if token and token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def dual_header_api_key(expected: str, authorization: str | None, x_api_key: str | None) -> None:
    token = _extract_bearer(authorization) or (x_api_key or None)
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
