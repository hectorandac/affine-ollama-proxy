from __future__ import annotations

import json

import pytest

from ollama_proxy.gateway import StreamChunk


def _auth_header():
    return {"Authorization": "Bearer test-key"}


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.asyncio
async def test_responses_non_stream(client, fake_gateway):
    fake_gateway.chat_response = {
        "choices": [{"message": {"content": "Hello from upstream"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    payload = {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == ["Hello from upstream"]
    assert fake_gateway.last_chat_payload["model"] == "test-base"


@pytest.mark.asyncio
async def test_responses_title_detection_switches_model(client, fake_gateway):
    fake_gateway.chat_response = {
        "choices": [{"message": {"content": "short title"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    payload = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": "Please create a title under 32 characters for this note",
            }
        ],
        "stream": False,
    }
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    assert response.status_code == 200
    assert fake_gateway.last_chat_payload["model"] == "test-title"


@pytest.mark.asyncio
async def test_responses_streaming_fallback_to_generate(client, fake_gateway):
    fake_gateway.raise_chat_not_found = True
    fake_gateway.generate_stream_chunks = [
        StreamChunk(kind="line", data=json.dumps({"response": "Hello"})),
        StreamChunk(kind="line", data=json.dumps({"response": " world"})),
    ]
    payload = {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    body = (await response.aread()).decode("utf-8")
    assert "response.output_text.delta" in body
    assert "Hello world" in body
    assert fake_gateway.last_generate_payload["model"] == "test-base"


@pytest.mark.asyncio
async def test_models_requires_valid_token_when_provided(client):
    response = await client.get("/v1/models", headers={"Authorization": "Bearer bad"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_models_success(client):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data[0]["object"] == "model"


@pytest.mark.asyncio
async def test_models_default_on_error(client, fake_gateway):
    fake_gateway.raise_models_error = True
    response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "test-base"


@pytest.mark.asyncio
async def test_embeddings_mock_fallback(client, fake_gateway):
    fake_gateway.raise_embeddings_not_found = True
    payload = {"model": "foo", "input": "hello"}
    response = await client.post("/v1/embeddings", json=payload, headers=_auth_header())
    assert response.status_code == 200
    embedding = response.json()["data"][0]["embedding"]
    assert len(embedding) == 1536


@pytest.mark.asyncio
async def test_embeddings_success_path(client, fake_gateway):
    fake_gateway.embeddings_response = {"embedding": [0.5, 0.4, 0.3]}
    payload = {"model": "foo", "input": ["hello", "world"]}
    response = await client.post("/v1/embeddings", json=payload, headers=_auth_header())
    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [0.5, 0.4, 0.3]


@pytest.mark.asyncio
async def test_embeddings_invalid_input(client):
    payload = {"model": "foo"}
    response = await client.post("/v1/embeddings", json=payload, headers=_auth_header())
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_embeddings_invalid_json(client):
    response = await client.post(
        "/v1/embeddings",
        data="not-json",
        headers={**_auth_header(), "Content-Type": "application/json"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_embeddings_invalid_list(client):
    payload = {"model": "foo", "input": []}
    response = await client.post("/v1/embeddings", json=payload, headers=_auth_header())
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_embeddings_upstream_error(client, fake_gateway):
    fake_gateway.raise_embeddings_error = True
    payload = {"model": "foo", "input": "hello"}
    response = await client.post("/v1/embeddings", json=payload, headers=_auth_header())
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_messages_non_stream(client, fake_gateway):
    fake_gateway.chat_response = {
        "choices": [{"message": {"content": "Anthropic reply"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    payload = {
        "model": "claude",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "system": "system prompt",
        "stream": False,
    }
    response = await client.post(
        "/v1/messages", json=payload, headers={"x-api-key": "test-key"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["text"] == "Anthropic reply"
    assert fake_gateway.last_chat_payload["model"] == "test-base"


@pytest.mark.asyncio
async def test_messages_streaming_fallback(client, fake_gateway):
    fake_gateway.raise_chat_not_found = True
    fake_gateway.generate_stream_chunks = [
        StreamChunk(kind="line", data=json.dumps({"response": "Hello"})),
        StreamChunk(kind="line", data=json.dumps({"response": " there"})),
    ]
    payload = {
        "model": "claude",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "stream": True,
    }
    response = await client.post(
        "/v1/messages", json=payload, headers={"x-api-key": "test-key"}
    )
    body = (await response.aread()).decode("utf-8")
    assert "content_block_delta" in body
    assert "Hello" in body
    assert " there" in body


@pytest.mark.asyncio
async def test_chat_completions_fallback_generate(client, fake_gateway):
    fake_gateway.raise_chat_not_found = True
    fake_gateway.generate_response = {
        "response": "Generated",
        "prompt_eval_count": 3,
        "eval_count": 2,
    }
    payload = {"model": "foo", "messages": [{"role": "user", "content": "yo"}]}
    response = await client.post("/v1/chat/completions", json=payload, headers=_auth_header())
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Generated"
    assert fake_gateway.last_generate_payload["model"] == "test-base"


@pytest.mark.asyncio
async def test_chat_completions_stream_passthrough(client, fake_gateway):
    fake_gateway.chat_stream_bytes = [b"data: chunk\n\n"]
    payload = {"model": "foo", "messages": [{"role": "user", "content": "stream"}], "stream": True}
    response = await client.post("/v1/chat/completions", json=payload, headers=_auth_header())
    body = await response.aread()
    assert b"chunk" in body


@pytest.mark.asyncio
async def test_responses_non_stream_fallback_generate(client, fake_gateway):
    fake_gateway.raise_chat_not_found = True
    fake_gateway.generate_response = {
        "response": "fallback text",
        "prompt_eval_count": 5,
        "eval_count": 7,
    }
    payload = {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    assert response.status_code == 200
    assert response.json()["output_text"][0] == "fallback text"


@pytest.mark.asyncio
async def test_responses_streaming_openai_path(client, fake_gateway):
    fake_gateway.chat_stream_chunks = [
        StreamChunk(
            kind="line",
            data=json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
        ),
        StreamChunk(
            kind="line",
            data=json.dumps(
                {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]}
            ),
        ),
    ]
    payload = {"model": "gpt-4o", "input": [{"role": "user", "content": "hi"}], "stream": True}
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    body = (await response.aread()).decode("utf-8")
    assert "Hello " in body and "world" in body


@pytest.mark.asyncio
async def test_messages_non_stream_fallback(client, fake_gateway):
    fake_gateway.raise_chat_not_found = True
    fake_gateway.generate_response = {"response": "Anthropic fallback", "prompt_eval_count": 1, "eval_count": 1}
    payload = {
        "model": "claude",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "stream": False,
    }
    response = await client.post(
        "/v1/messages", json=payload, headers={"x-api-key": "test-key"}
    )
    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "Anthropic fallback"


@pytest.mark.asyncio
async def test_chat_completions_streaming_fallback(client, fake_gateway):
    fake_gateway.raise_chat_not_found = True
    fake_gateway.generate_stream_chunks = [
        StreamChunk(kind="line", data=json.dumps({"response": "partial"}))
    ]
    payload = {"model": "foo", "messages": [{"role": "user", "content": "stream"}], "stream": True}
    response = await client.post("/v1/chat/completions", json=payload, headers=_auth_header())
    body = await response.aread()
    assert b"partial" in body


@pytest.mark.asyncio
async def test_chat_completions_invalid_json(client):
    response = await client.post(
        "/v1/chat/completions",
        data="not-json",
        headers={**_auth_header(), "Content-Type": "application/json"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_respects_options(client, fake_gateway):
    payload = {
        "model": "foo",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.9,
        "max_tokens": 42,
    }
    await client.post("/v1/chat/completions", json=payload, headers=_auth_header())
    assert fake_gateway.last_chat_payload["temperature"] == 0.9
    assert fake_gateway.last_chat_payload["max_tokens"] == 42


@pytest.mark.asyncio
async def test_chat_completions_non_stream_success(client, fake_gateway):
    fake_gateway.chat_response = {"choices": [{"message": {"content": "ok"}}]}
    payload = {"model": "foo", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    response = await client.post("/v1/chat/completions", json=payload, headers=_auth_header())
    assert response.status_code == 200
    assert response.json()["model"] == "foo"


@pytest.mark.asyncio
async def test_responses_invalid_json(client):
    response = await client.post(
        "/v1/responses",
        data="not-json",
        headers={**_auth_header(), "Content-Type": "application/json"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_responses_title_truncate_non_stream(client, fake_gateway):
    fake_gateway.chat_response = {
        "choices": [{"message": {"content": "x" * 60}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    payload = {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Please create a title under 32 characters"}],
        "stream": False,
    }
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    assert len(response.json()["output_text"][0]) <= 32


@pytest.mark.asyncio
async def test_responses_title_truncate_fallback(client, fake_gateway):
    fake_gateway.raise_chat_not_found = True
    fake_gateway.generate_response = {"response": "y" * 80, "prompt_eval_count": 0, "eval_count": 0}
    payload = {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Summarize title in 32 characters"}],
        "stream": False,
    }
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    assert len(response.json()["output_text"][0]) == 32


@pytest.mark.asyncio
async def test_responses_streaming_truncate_limit(client, fake_gateway):
    fake_gateway.chat_stream_chunks = [
        StreamChunk(
            kind="line",
            data=json.dumps({"choices": [{"delta": {"content": "z" * 50}}]}),
        )
    ]
    payload = {
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Title under 32 characters"}],
        "stream": True,
    }
    response = await client.post("/v1/responses", json=payload, headers=_auth_header())
    body = (await response.aread()).decode("utf-8")
    assert "z" * 40 not in body


@pytest.mark.asyncio
async def test_messages_invalid_json(client):
    response = await client.post(
        "/v1/messages",
        data="not-json",
        headers={"x-api-key": "test-key", "Content-Type": "application/json"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_messages_temperature_max_tokens(client, fake_gateway):
    payload = {
        "model": "claude",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "temperature": 0.5,
        "max_tokens": 128,
    }
    await client.post("/v1/messages", json=payload, headers={"x-api-key": "test-key"})
    assert fake_gateway.last_chat_payload["temperature"] == 0.5
    assert fake_gateway.last_chat_payload["max_tokens"] == 128


@pytest.mark.asyncio
async def test_messages_streaming_openai_path(client, fake_gateway):
    fake_gateway.chat_stream_chunks = [
        StreamChunk(
            kind="line",
            data=json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}),
        ),
        StreamChunk(
            kind="line",
            data=json.dumps({"choices": [{"delta": {"content": " friend"}, "finish_reason": "stop"}]}),
        ),
    ]
    payload = {
        "model": "claude",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "stream": True,
    }
    response = await client.post(
        "/v1/messages", json=payload, headers={"x-api-key": "test-key"}
    )
    body = (await response.aread()).decode("utf-8")
    assert "Hello" in body
    assert " friend" in body
