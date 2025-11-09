from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from ..auth import require_api_key
from ..config import Settings
from ..dependencies import gateway_dependency, settings_dependency
from ..gateway import UpstreamClient, UpstreamError, UpstreamNotFoundError

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(
    request: Request,
    authorization: str = Header(default=""),
    settings: Settings = Depends(settings_dependency),
    upstream: UpstreamClient = Depends(gateway_dependency),
) -> dict:
    require_api_key(settings.api_key, authorization)
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    inputs = payload.get("input")
    if inputs is None:
        raise HTTPException(status_code=400, detail="Missing input")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not inputs:
        raise HTTPException(status_code=400, detail="Invalid input")

    model = payload.get("model", settings.base_model)
    ollama_payload = {
        "model": model,
        "prompt": inputs[0] if len(inputs) == 1 else inputs,
    }

    try:
        data = await upstream.embeddings(ollama_payload)
        embedding = data.get("embedding", [])
    except UpstreamNotFoundError:
        embedding = [0.0] * 1536
    except UpstreamError as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}") from exc

    response_data = [
        {"object": "embedding", "index": idx, "embedding": embedding if idx == 0 else []}
        for idx in range(len(inputs))
    ]
    return {
        "object": "list",
        "data": response_data,
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
