from __future__ import annotations

import time
from fastapi import APIRouter, Depends, Header

from ..auth import optional_api_key
from ..config import Settings
from ..dependencies import gateway_dependency, settings_dependency
from ..gateway import UpstreamClient, UpstreamError

router = APIRouter()


@router.get("/v1/models")
async def list_models(
    authorization: str = Header(default=""),
    settings: Settings = Depends(settings_dependency),
    upstream: UpstreamClient = Depends(gateway_dependency),
) -> dict:
    optional_api_key(settings.api_key, authorization)
    default_response = {
        "object": "list",
        "data": [
            {
                "id": settings.base_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
            }
        ],
    }
    try:
        ollama_models = await upstream.list_models()
    except UpstreamError:
        return default_response

    data = []
    for model in ollama_models.get("models", []):
        data.append(
            {
                "id": model.get("name"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
            }
        )
    return {"object": "list", "data": data or default_response["data"]}
