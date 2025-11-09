from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator, Literal, Protocol

import httpx

from .config import Settings


class UpstreamError(RuntimeError):
    """Base exception for upstream communication issues."""


class UpstreamNotFoundError(UpstreamError):
    """Raised when the upstream reports a missing endpoint (HTTP 404)."""


@dataclass
class StreamChunk:
    kind: Literal["json", "line"]
    data: bytes | str


class UpstreamClient(Protocol):
    async def chat_completions(self, payload: dict) -> dict: ...

    async def stream_chat_chunks(self, payload: dict) -> AsyncGenerator[StreamChunk, None]: ...

    async def stream_chat_bytes(self, payload: dict) -> AsyncGenerator[bytes, None]: ...

    async def generate(self, payload: dict) -> dict: ...

    async def stream_generate_chunks(self, payload: dict) -> AsyncGenerator[StreamChunk, None]: ...

    async def list_models(self) -> dict: ...

    async def embeddings(self, payload: dict) -> dict: ...


class HttpUpstreamClient(UpstreamClient):
    """httpx-based implementation used in production and Docker."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._settings.httpx_timeout)

    def _url(self, path: str) -> str:
        return f"{self._settings.upstream.rstrip('/')}{path}"

    async def _post_json(self, path: str, payload: dict) -> dict:
        async with self._client() as client:
            try:
                response = await client.post(self._url(path), json=payload)
                if response.status_code == 404:
                    raise UpstreamNotFoundError(f"{path} not found")
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:  # pragma: no cover - delegated to helper
                if exc.response is not None and exc.response.status_code == 404:
                    raise UpstreamNotFoundError(path) from exc
                raise UpstreamError(str(exc)) from exc
            return response.json()

    async def _get_json(self, path: str) -> dict:
        async with self._client() as client:
            try:
                response = await client.get(self._url(path))
                if response.status_code == 404:
                    raise UpstreamNotFoundError(f"{path} not found")
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:  # pragma: no cover - delegated to helper
                if exc.response is not None and exc.response.status_code == 404:
                    raise UpstreamNotFoundError(path) from exc
                raise UpstreamError(str(exc)) from exc
            return response.json()

    async def _stream_chunks(self, path: str, payload: dict) -> AsyncGenerator[StreamChunk, None]:
        async with self._client() as client:
            try:
                async with client.stream("POST", self._url(path), json=payload) as response:
                    if response.status_code == 404:
                        raise UpstreamNotFoundError(f"{path} not found")
                    content_type = (response.headers.get("Content-Type") or "").lower()
                    if "application/json" in content_type:
                        body = await response.aread()
                        yield StreamChunk(kind="json", data=body)
                        return
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield StreamChunk(kind="line", data=line)
            except httpx.HTTPStatusError as exc:  # pragma: no cover - delegated to helper
                if exc.response is not None and exc.response.status_code == 404:
                    raise UpstreamNotFoundError(path) from exc
                raise UpstreamError(str(exc)) from exc

    async def _stream_bytes(self, path: str, payload: dict) -> AsyncGenerator[bytes, None]:
        async with self._client() as client:
            try:
                async with client.stream("POST", self._url(path), json=payload) as response:
                    if response.status_code == 404:
                        raise UpstreamNotFoundError(f"{path} not found")
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
            except httpx.HTTPStatusError as exc:  # pragma: no cover - delegated to helper
                if exc.response is not None and exc.response.status_code == 404:
                    raise UpstreamNotFoundError(path) from exc
                raise UpstreamError(str(exc)) from exc

    async def chat_completions(self, payload: dict) -> dict:
        return await self._post_json("/v1/chat/completions", payload)

    async def stream_chat_chunks(self, payload: dict) -> AsyncGenerator[StreamChunk, None]:
        async for chunk in self._stream_chunks("/v1/chat/completions", payload):
            yield chunk

    async def stream_chat_bytes(self, payload: dict) -> AsyncGenerator[bytes, None]:
        async for chunk in self._stream_bytes("/v1/chat/completions", payload):
            yield chunk

    async def generate(self, payload: dict) -> dict:
        return await self._post_json("/api/generate", payload)

    async def stream_generate_chunks(self, payload: dict) -> AsyncGenerator[StreamChunk, None]:
        async for chunk in self._stream_chunks("/api/generate", payload):
            yield chunk

    async def list_models(self) -> dict:
        return await self._get_json("/api/tags")

    async def embeddings(self, payload: dict) -> dict:
        return await self._post_json("/api/embeddings", payload)
