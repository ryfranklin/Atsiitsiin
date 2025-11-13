"""Atsiitsʼiin – Snowflake-backed agentic second brain."""

from .agents import AtsiiitsiinAgent
from .config import AtsiiitsiinConfig, get_snowflake_connection
from .memory import (
    Embedder,
    SnowflakeMemoryStore,
    ingest_note,
    search_memory,
    simple_chunk,
)

__all__ = [
    "AtsiiitsiinAgent",
    "AtsiiitsiinConfig",
    "Embedder",
    "SnowflakeMemoryStore",
    "get_snowflake_connection",
    "ingest_note",
    "search_memory",
    "simple_chunk",
]
