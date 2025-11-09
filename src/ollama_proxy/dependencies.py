from __future__ import annotations

from fastapi import Depends

from .config import Settings, get_settings
from .gateway import HttpUpstreamClient, UpstreamClient


def settings_dependency() -> Settings:
    return get_settings()


def gateway_dependency(settings: Settings = Depends(settings_dependency)) -> UpstreamClient:
    return HttpUpstreamClient(settings=settings)
