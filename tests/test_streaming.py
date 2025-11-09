from __future__ import annotations

import json
import pytest

from ollama_proxy.gateway import StreamChunk
from ollama_proxy.services.streaming import ollama_text_stream, openai_text_stream
from tests.fakes import FakeGateway


@pytest.mark.asyncio
async def test_openai_text_stream_handles_json_payload():
    gateway = FakeGateway()
    gateway.chat_stream_chunks = [
        StreamChunk(
            kind="json",
            data=json.dumps({"choices": [{"message": {"content": "json chunk"}}]}).encode("utf-8"),
        )
    ]
    body = {"model": "test-base", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    pieces = []
    async for piece in openai_text_stream(body, gateway):
        pieces.append(piece)
    assert pieces == ["json chunk"]


@pytest.mark.asyncio
async def test_openai_text_stream_handles_sse_lines():
    gateway = FakeGateway()
    gateway.chat_stream_chunks = [
        StreamChunk(
            kind="line",
            data='data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}' ,
        ),
        StreamChunk(
            kind="line",
            data='data: {"choices": [{"delta": {"content": " there"}, "finish_reason": "stop"}]}',
        ),
    ]
    body = {"model": "test-base", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    pieces = []
    async for piece in openai_text_stream(body, gateway):
        pieces.append(piece)
    assert pieces == ["Hello", " there"]


@pytest.mark.asyncio
async def test_openai_text_stream_handles_done_and_invalid_lines():
    gateway = FakeGateway()
    gateway.chat_stream_chunks = [
        StreamChunk(kind="line", data="data: not-json"),
        StreamChunk(kind="line", data="data: [DONE]"),
    ]
    body = {"model": "test-base", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    pieces = []
    async for piece in openai_text_stream(body, gateway):
        pieces.append(piece)
    assert pieces == []


@pytest.mark.asyncio
async def test_ollama_text_stream_parses_json_lines():
    gateway = FakeGateway()
    gateway.generate_stream_chunks = [
        StreamChunk(kind="line", data=json.dumps({"response": "Hello"})),
        StreamChunk(kind="line", data=json.dumps({"response": " there"})),
    ]
    body = {"model": "test-base", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    pieces = []
    async for piece in ollama_text_stream(body, gateway):
        pieces.append(piece)
    assert "".join(pieces) == "Hello there"


@pytest.mark.asyncio
async def test_ollama_text_stream_skips_invalid_json():
    gateway = FakeGateway()
    gateway.generate_stream_chunks = [
        StreamChunk(kind="line", data="not-json"),
        StreamChunk(kind="line", data=json.dumps({"response": "ok"})),
    ]
    body = {"model": "test-base", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    pieces = []
    async for piece in ollama_text_stream(body, gateway):
        pieces.append(piece)
    assert pieces == ["ok"]
