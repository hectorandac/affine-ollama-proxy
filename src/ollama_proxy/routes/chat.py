from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import require_api_key
from ..config import Settings
from ..dependencies import gateway_dependency, settings_dependency
from ..gateway import UpstreamClient, UpstreamNotFoundError
from ..services.streaming import ollama_text_stream
from ..utils import build_generate_payload

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str = Header(default=""),
    settings: Settings = Depends(settings_dependency),
    upstream: UpstreamClient = Depends(gateway_dependency),
):
    require_api_key(settings.api_key, authorization)
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    original_model = payload.get("model", settings.base_model)
    messages = payload.get("messages", [])
    stream = payload.get("stream", False)
    upstream_body = {
        "model": settings.base_model,
        "messages": messages,
        "stream": stream,
    }
    if payload.get("temperature") is not None:
        upstream_body["temperature"] = payload["temperature"]
    if payload.get("max_tokens") is not None:
        upstream_body["max_tokens"] = payload["max_tokens"]

    if stream:
        async def forward_stream():
            try:
                async for chunk in upstream.stream_chat_bytes(upstream_body):
                    yield chunk
            except UpstreamNotFoundError:
                async for text in ollama_text_stream(upstream_body, upstream):
                    yield f"data: {text}\n\n".encode("utf-8")

        return StreamingResponse(
            forward_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await upstream.chat_completions(upstream_body)
        response["model"] = original_model
        return JSONResponse(content=response)
    except UpstreamNotFoundError:
        gen_payload = build_generate_payload(upstream_body)
        gen_payload["stream"] = False
        ollama_response = await upstream.generate(gen_payload)
        content = ollama_response.get("response", "")
        openai_format = {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": original_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "completion_tokens": ollama_response.get("eval_count", 0),
                "total_tokens": ollama_response.get("prompt_eval_count", 0)
                + ollama_response.get("eval_count", 0),
            },
        }
        return JSONResponse(content=openai_format)
