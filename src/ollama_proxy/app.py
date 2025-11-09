from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routes import chat, embeddings, health, messages, models, responses


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Ollama Affine Proxy")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(embeddings.router)
    app.include_router(responses.router)
    app.include_router(messages.router)
    app.include_router(chat.router)
    return app


app = create_app()
