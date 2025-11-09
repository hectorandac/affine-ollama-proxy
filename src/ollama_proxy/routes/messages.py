from __future__ import annotations

import json
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import dual_header_api_key
from ..config import Settings
from ..dependencies import gateway_dependency, settings_dependency
from ..gateway import UpstreamClient, UpstreamNotFoundError
from ..services.streaming import ollama_text_stream, openai_text_stream
from ..utils import (
    anthropic_messages_to_openai,
    build_generate_payload,
    openai_json_to_text,
)

router = APIRouter()


def _anthropic_message(content: str, model: str, usage: dict) -> dict:
    return {
        "id": f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


@router.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    authorization: str = Header(default=""),
    x_api_key: str = Header(default=""),
    settings: Settings = Depends(settings_dependency),
    upstream: UpstreamClient = Depends(gateway_dependency),
):
    dual_header_api_key(settings.api_key, authorization, x_api_key)
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    original_model = payload.get("model", settings.base_model)
    messages = anthropic_messages_to_openai(payload.get("messages", []), payload.get("system"))
    temperature = payload.get("temperature")
    max_tokens = payload.get("max_tokens")
    stream = payload.get("stream", False)

    upstream_body = {
        "model": settings.base_model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        upstream_body["temperature"] = temperature
    if max_tokens is not None:
        upstream_body["max_tokens"] = max_tokens

    if stream:
        async def content_source() -> AsyncGenerator[str, None]:
            try:
                async for piece in openai_text_stream(upstream_body, upstream):
                    yield piece
            except UpstreamNotFoundError:
                async for piece in ollama_text_stream(upstream_body, upstream):
                    yield piece

        async def stream_response():
            message_id = f"msg_{uuid4().hex}"
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': original_model, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            full_content = ""
            async for piece in content_source():
                full_content += piece
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': piece}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        upstream_response = await upstream.chat_completions(upstream_body)
        content = openai_json_to_text(upstream_response)
        usage = {
            "input_tokens": upstream_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": upstream_response.get("usage", {}).get("completion_tokens", 0),
        }
        return JSONResponse(content=_anthropic_message(content, original_model, usage))
    except UpstreamNotFoundError:
        payload = build_generate_payload(upstream_body)
        payload["stream"] = False
        ollama_response = await upstream.generate(payload)
        content = ollama_response.get("response", "")
        usage = {
            "input_tokens": ollama_response.get("prompt_eval_count", 0),
            "output_tokens": ollama_response.get("eval_count", 0),
        }
        return JSONResponse(content=_anthropic_message(content, original_model, usage))
