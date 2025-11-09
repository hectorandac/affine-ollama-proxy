from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict

from ..gateway import UpstreamClient
from ..utils import build_generate_payload, openai_json_to_text


async def openai_text_stream(
    body: Dict[str, Any],
    upstream: UpstreamClient,
) -> AsyncGenerator[str, None]:
    async for chunk in upstream.stream_chat_chunks(dict(body, stream=True)):
        if chunk.kind == "json":
            obj = json.loads(chunk.data.decode("utf-8", "ignore"))
            text = openai_json_to_text(obj)
            if text:
                yield text
            return
        line = chunk.data
        if isinstance(line, bytes):
            line = line.decode("utf-8", "ignore")
        if not line:
            continue
        normalized = line[6:] if line.startswith("data: ") else line
        if normalized.strip() == "[DONE]":
            break
        try:
            obj = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        choice = (obj.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece:
            yield piece
        if choice.get("finish_reason"):
            break


async def ollama_text_stream(
    body: Dict[str, Any],
    upstream: UpstreamClient,
) -> AsyncGenerator[str, None]:
    payload = build_generate_payload(dict(body, stream=True))
    async for chunk in upstream.stream_generate_chunks(payload):
        line = chunk.data.decode("utf-8", "ignore") if isinstance(chunk.data, bytes) else chunk.data
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        piece = obj.get("response")
        if piece:
            yield piece
