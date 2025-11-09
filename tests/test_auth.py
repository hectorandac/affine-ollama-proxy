from __future__ import annotations

import pytest
from fastapi import HTTPException

from ollama_proxy.auth import dual_header_api_key, optional_api_key, require_api_key


def test_require_api_key_rejects_invalid():
    with pytest.raises(HTTPException):
        require_api_key("expected", "Bearer nope")


def test_optional_api_key_allows_missing_and_rejects_invalid():
    optional_api_key("expected", None)
    with pytest.raises(HTTPException):
        optional_api_key("expected", "Bearer bad")


def test_dual_header_api_key_supports_x_api_key():
    dual_header_api_key("token", None, "token")
    with pytest.raises(HTTPException):
        dual_header_api_key("token", "Bearer nope", None)
