from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import require_api_key
from ..config import Settings
from ..dependencies import gateway_dependency, settings_dependency
from ..gateway import UpstreamClient, UpstreamNotFoundError
from ..services.streaming import ollama_text_stream, openai_text_stream
from ..utils import (
    build_generate_payload,
    detect_title_generation,
    final_responses_object,
    openai_json_to_text,
    openai_usage_to_responses_usage,
    responses_to_chat_body,
)

router = APIRouter()


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_responses(
    body: Dict[str, Any],
    upstream: UpstreamClient,
    truncate_title: bool,
) -> AsyncGenerator[str, None]:
    response_id = f"resp_{uuid4().hex}"
    item_id = f"item_{uuid4().hex}"
    output_index = 0
    content_index = 0
    full_content = ""
    first_delta = True
    created_ts = int(time.time())

    async def try_openai() -> AsyncGenerator[str, None]:
        async for text in openai_text_stream(body, upstream):
            yield text

    async def try_generate() -> AsyncGenerator[str, None]:
        async for text in ollama_text_stream(body, upstream):
            yield text

    async def content_source() -> AsyncGenerator[str, None]:
        try:
            async for piece in try_openai():
                yield piece
        except UpstreamNotFoundError:
            async for piece in try_generate():
                yield piece

    # Initial response events
    yield _event(
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created": created_ts,
                "status": "in_progress",
            },
        }
    )
    yield _event(
        {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "object": "response.output_item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
    )

    async for piece in content_source():
        if not piece:
            continue
        if truncate_title:
            remaining = 32 - len(full_content)
            if remaining <= 0:
                break
            piece = piece[:remaining]
            if not piece:
                break
        if first_delta:
            yield _event(
                {
                    "type": "response.content_part.added",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": ""},
                }
            )
            first_delta = False
        yield _event(
            {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": piece,
            }
        )
        full_content += piece

    yield _event(
        {
            "type": "response.content_part.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": {"type": "output_text", "text": full_content},
        }
    )
    yield _event(
        {
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "object": "response.output_item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": full_content}],
            },
        }
    )
    yield _event(
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created": created_ts,
                "status": "completed",
                "output_text": [full_content],
                "output": [
                    {
                        "id": item_id,
                        "object": "response.output_item",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": full_content}],
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        }
    )


async def call_chat_or_fallback_responses_final(
    body: Dict[str, Any],
    upstream: UpstreamClient,
    truncate_title: bool,
) -> Dict[str, Any]:
    body_local = dict(body)
    body_local["stream"] = False
    try:
        upstream_response = await upstream.chat_completions(body_local)
    except UpstreamNotFoundError:
        payload = build_generate_payload(body_local)
        ollama_response = await upstream.generate(payload)
        text = ollama_response.get("response", "")
        if truncate_title and len(text) > 32:
            text = text[:32].strip()
        usage = {
            "input_tokens": ollama_response.get("prompt_eval_count", 0),
            "output_tokens": ollama_response.get("eval_count", 0),
            "total_tokens": ollama_response.get("prompt_eval_count", 0)
            + ollama_response.get("eval_count", 0),
        }
        return final_responses_object(body_local.get("original_model", body_local["model"]), text, usage)

    text = openai_json_to_text(upstream_response)
    if truncate_title and len(text) > 32:
        text = text[:32].strip()
    usage = openai_usage_to_responses_usage(upstream_response)
    return final_responses_object(body_local.get("original_model", body_local["model"]), text, usage)


@router.post("/v1/responses")
async def responses_endpoint(
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

    body = responses_to_chat_body(payload, settings.base_model)
    is_title = detect_title_generation(payload.get("input"))
    if is_title:
        body["model"] = settings.title_model

    stream = bool(payload.get("stream", True))
    if stream:
        return StreamingResponse(
            stream_responses(body, upstream, truncate_title=is_title),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    responses_object = await call_chat_or_fallback_responses_final(
        body, upstream, truncate_title=is_title
    )
    return JSONResponse(content=responses_object)
