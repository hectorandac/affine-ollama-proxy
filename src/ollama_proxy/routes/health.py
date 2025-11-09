from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import Settings
from ..dependencies import settings_dependency

router = APIRouter()


@router.get("/health")
async def health(settings: Settings = Depends(settings_dependency)) -> dict:
    return {"ok": True, "upstream": settings.upstream}
