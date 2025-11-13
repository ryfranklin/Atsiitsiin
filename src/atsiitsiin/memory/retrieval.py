from ..config import AtsiiitsiinConfig, get_snowflake_connection
from .embeddings import Embedder
from .store import SnowflakeMemoryStore


def search_memory(
    query: str, cfg: AtsiiitsiinConfig, k: int = 8
) -> list[dict]:
    embedder = Embedder(cfg.embedding_model)
    qv = embedder.embed_texts([query])[0]
    with get_snowflake_connection(
        cfg.snowflake_env_file
    ).session_context() as session:
        store = SnowflakeMemoryStore(session, cfg.embedding_dim)
        rows = store.similarity_search(qv, k)
        return [dict(row) for row in rows]


def list_notes(
    cfg: AtsiiitsiinConfig, limit: int = 20, offset: int = 0
) -> list[dict]:
    with get_snowflake_connection(
        cfg.snowflake_env_file
    ).session_context() as session:
        store = SnowflakeMemoryStore(session, cfg.embedding_dim)
        return store.list_notes(limit=limit, offset=offset)


def get_note(
    note_id: str,
    cfg: AtsiiitsiinConfig,
    include_chunks: bool = False,
) -> dict | None:
    with get_snowflake_connection(
        cfg.snowflake_env_file
    ).session_context() as session:
        store = SnowflakeMemoryStore(session, cfg.embedding_dim)
        result = store.get_note(note_id, include_chunks=include_chunks)
        return dict(result) if result is not None else None


def sentiment_summary(cfg: AtsiiitsiinConfig) -> dict[str, int | float | None]:
    with get_snowflake_connection(
        cfg.snowflake_env_file
    ).session_context() as session:
        result = session.sql(
            """
            SELECT
                COUNT(*) AS total_notes,
                COUNT_IF(CONTEXT_LABEL = 'work') AS work_notes,
                COUNT_IF(CONTEXT_LABEL = 'personal') AS personal_notes,
                COUNT_IF(CONTEXT_LABEL = 'leisure') AS leisure_notes,
                COUNT_IF(CONTEXT_LABEL = 'mixed') AS mixed_notes,
                COUNT_IF(
                    CONTEXT_LABEL NOT IN ('work', 'personal', 'leisure', 'mixed')
                    AND CONTEXT_LABEL IS NOT NULL
                ) AS unknown_notes,
                COUNT_IF(CONTEXT_ANALYZED_AT IS NULL) AS pending_notes,
                AVG(CONTEXT_CONFIDENCE) AS average_context_confidence,
                MAX(CONTEXT_ANALYZED_AT) AS last_analyzed_at
            FROM NOTES
            """
        ).collect()
        if not result:
            return {}
        row = result[0]
        if hasattr(row, "as_dict"):
            return row.as_dict()  # type: ignore[return-value]
        return dict(row)


def count_notes(cfg: AtsiiitsiinConfig) -> int:
    with get_snowflake_connection(
        cfg.snowflake_env_file
    ).session_context() as session:
        store = SnowflakeMemoryStore(session, cfg.embedding_dim)
        return store.count_notes()
