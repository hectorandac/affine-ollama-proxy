from __future__ import annotations

from ollama_proxy.config import Settings, _comma_separated
from ollama_proxy.dependencies import gateway_dependency, settings_dependency
from ollama_proxy.gateway import HttpUpstreamClient


def test_settings_dependency_returns_cached_object():
    first = settings_dependency()
    second = settings_dependency()
    assert first is second


def test_gateway_dependency_builds_client():
    settings = Settings(upstream="http://localhost:11434")
    client = gateway_dependency(settings)
    assert isinstance(client, HttpUpstreamClient)


def test_comma_separated_defaults_and_timeout():
    assert _comma_separated(None) == ["*"]
    settings = Settings()
    timeout = settings.httpx_timeout
    assert timeout.read == settings.timeout
