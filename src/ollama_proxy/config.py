from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

import httpx
from dotenv import load_dotenv

load_dotenv()


def _comma_separated(value: str | None) -> List[str]:
    if not value:
        return ["*"]
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass
class Settings:
    """Runtime configuration loaded from environment variables."""

    upstream: str = os.getenv("UPSTREAM", "http://10.147.20.68:11434")
    api_key: str = os.getenv("PROXY_API_KEY", "ollama-local")
    base_model: str = os.getenv("PROXY_BASE_MODEL", "gpt-oss:20b")
    title_model: str = os.getenv("PROXY_TITLE_MODEL", "granite3.1-moe:1b")
    timeout: float = float(os.getenv("PROXY_TIMEOUT", "300.0"))
    connect_timeout: float = float(os.getenv("PROXY_CONNECT_TIMEOUT", "10.0"))
    cors_allow_origins: List[str] = field(
        default_factory=lambda: _comma_separated(os.getenv("PROXY_CORS_ALLOW_ORIGINS"))
    )

    @property
    def httpx_timeout(self) -> httpx.Timeout:
        """Build a reusable httpx timeout configuration."""
        return httpx.Timeout(self.timeout, connect=self.connect_timeout, read=self.timeout)


@lru_cache
def get_settings() -> Settings:
    return Settings()
