from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ollama_proxy.app import create_app
from ollama_proxy.config import Settings
from ollama_proxy.dependencies import gateway_dependency, settings_dependency
from tests.fakes import FakeGateway


@pytest.fixture
def settings() -> Settings:
    return Settings(
        upstream="http://upstream",
        api_key="test-key",
        base_model="test-base",
        title_model="test-title",
        timeout=30.0,
        connect_timeout=5.0,
        cors_allow_origins=["*"],
    )


@pytest.fixture
def fake_gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
async def client(settings: Settings, fake_gateway: FakeGateway):
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: settings
    app.dependency_overrides[gateway_dependency] = lambda: fake_gateway
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
