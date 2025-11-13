from pydantic import BaseModel, Field

from ..integrations.snowflake import SnowflakeConfig, SnowflakeConnection


class AtsiiitsiinConfig(BaseModel):
    snowflake_env_file: str = Field(default=".env")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dim: int = Field(default=1536)
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)
    # LLM configuration
    llm_model: str = Field(default="openai/gpt-4o")
    llm_max_tokens: int = Field(default=1024)
    llm_temperature: float = Field(default=0.7)


def get_snowflake_connection(env_file: str = ".env") -> SnowflakeConnection:
    """Get a Snowflake connection instance from an environment file.

    Args:
        env_file: Path to the .env file containing Snowflake credentials

    Returns:
        SnowflakeConnection instance
    """
    cfg = SnowflakeConfig.from_env_file(env_file)
    return SnowflakeConnection(cfg)
