import os
from functools import lru_cache

from pydantic import BaseModel, Field

from atsiitsiin.config import AtsiiitsiinConfig


class BackendSettings(BaseModel):
    """Backend runtime configuration for the FastAPI application."""

    environment: str = Field(default="development")
    version: str = Field(default="0.1.0")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


@lru_cache(maxsize=1)
def get_backend_settings() -> BackendSettings:
    """Load backend-specific settings from environment variables."""
    environment = os.getenv("ATSIIITSIN_ENV", "development")
    version = os.getenv("ATSIIITSIN_API_VERSION", "0.1.0")
    origins_raw = os.getenv("ATSIIITSIN_CORS_ORIGINS")

    if origins_raw:
        origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]
    else:
        origins = ["*"]

    return BackendSettings(
        environment=environment,
        version=version,
        cors_origins=origins or ["*"],
    )


@lru_cache(maxsize=1)
def get_memory_config() -> AtsiiitsiinConfig:
    """Return the shared Atsiitsʼiin configuration instance."""
    env_file = os.getenv("ATSIIITSIN_ENV_FILE", ".env")
    return AtsiiitsiinConfig(snowflake_env_file=env_file)

