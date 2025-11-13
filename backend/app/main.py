from __future__ import annotations

import logging
from collections.abc import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import health, notes
from .core.config import get_backend_settings


def create_app() -> FastAPI:
    settings = get_backend_settings()
    application = FastAPI(
        title="Atsiitsʼiin API",
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    _configure_logging(settings.environment)
    _configure_cors(application, settings.cors_origins)
    _include_routes(application)

    return application


def _configure_logging(environment: str) -> None:
    logging.basicConfig(
        level=logging.INFO if environment == "production" else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    for noisy_logger in (
        "snowflake.connector",
        "snowflake.snowpark",
        "botocore",
        "boto3",
        "urllib3.connectionpool",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _configure_cors(application: FastAPI, allowed_origins: Sequence[str]) -> None:
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _include_routes(application: FastAPI) -> None:
    application.include_router(health.router)
    application.include_router(notes.router)


app = create_app()

