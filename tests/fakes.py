from __future__ import annotations

from typing import AsyncGenerator, List

from ollama_proxy.gateway import (
    StreamChunk,
    UpstreamClient,
    UpstreamError,
    UpstreamNotFoundError,
)


class FakeGateway(UpstreamClient):
    def __init__(self) -> None:
        self.chat_response: dict = {"choices": [{"message": {"content": "hello"}}]}
        self.generate_response: dict = {"response": "fallback"}
        self.embeddings_response: dict = {"embedding": [0.1, 0.2]}
        self.models_response: dict = {"models": [{"name": "fake"}]}
        self.raise_chat_not_found = False
        self.raise_embeddings_not_found = False
        self.raise_embeddings_error = False
        self.raise_models_error = False
        self.chat_stream_chunks: List[StreamChunk] = []
        self.generate_stream_chunks: List[StreamChunk] = []
        self.chat_stream_bytes: List[bytes] = []
        self.last_chat_payload: dict | None = None
        self.last_generate_payload: dict | None = None
        self.last_embeddings_payload: dict | None = None

    async def chat_completions(self, payload: dict) -> dict:
        self.last_chat_payload = payload
        if self.raise_chat_not_found:
            raise UpstreamNotFoundError("chat")
        return self.chat_response

    async def stream_chat_chunks(self, payload: dict) -> AsyncGenerator[StreamChunk, None]:
        self.last_chat_payload = payload
        if self.raise_chat_not_found:
            raise UpstreamNotFoundError("chat")
        for chunk in self.chat_stream_chunks:
            yield chunk

    async def stream_chat_bytes(self, payload: dict) -> AsyncGenerator[bytes, None]:
        self.last_chat_payload = payload
        if self.raise_chat_not_found:
            raise UpstreamNotFoundError("chat")
        for chunk in self.chat_stream_bytes:
            yield chunk

    async def generate(self, payload: dict) -> dict:
        self.last_generate_payload = payload
        return self.generate_response

    async def stream_generate_chunks(self, payload: dict) -> AsyncGenerator[StreamChunk, None]:
        self.last_generate_payload = payload
        for chunk in self.generate_stream_chunks:
            yield chunk

    async def list_models(self) -> dict:
        if self.raise_models_error:
            raise UpstreamError("models")
        return self.models_response

    async def embeddings(self, payload: dict) -> dict:
        self.last_embeddings_payload = payload
        if self.raise_embeddings_not_found:
            raise UpstreamNotFoundError("embeddings")
        if self.raise_embeddings_error:
            raise UpstreamError("boom")
        return self.embeddings_response
