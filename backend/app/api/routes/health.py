from __future__ import annotations

from fastapi import APIRouter

from ...core.config import get_backend_settings, get_memory_config

router = APIRouter(tags=["system"])


@router.get("/healthz", summary="Health check")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config", summary="Runtime configuration snapshot")
async def config() -> dict[str, object]:
    backend_settings = get_backend_settings()
    memory_config = get_memory_config()
    return {
        "environment": backend_settings.environment,
        "version": backend_settings.version,
        "cors_origins": backend_settings.cors_origins,
        "embedding_model": memory_config.embedding_model,
        "embedding_dim": memory_config.embedding_dim,
        "chunk_size": memory_config.chunk_size,
        "chunk_overlap": memory_config.chunk_overlap,
        "llm_model": memory_config.llm_model,
        "llm_max_tokens": memory_config.llm_max_tokens,
        "llm_temperature": memory_config.llm_temperature,
    }

