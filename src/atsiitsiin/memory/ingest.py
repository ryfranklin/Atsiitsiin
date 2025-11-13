from ..config import AtsiiitsiinConfig, get_snowflake_connection
from .embeddings import Embedder
from .store import SnowflakeMemoryStore


def simple_chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    if size <= 0:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        next_i = end - overlap
        i = next_i if next_i > i else end
    return chunks


def ingest_note(
    user_id: str,
    source: str,
    title: str,
    content: str,
    cfg: AtsiiitsiinConfig,
) -> str:
    chunks = simple_chunk(content, cfg.chunk_size, cfg.chunk_overlap)
    embedder = Embedder(cfg.embedding_model)
    vectors = embedder.embed_texts(chunks)

    with get_snowflake_connection(
        cfg.snowflake_env_file
    ).session_context() as session:
        store = SnowflakeMemoryStore(session, cfg.embedding_dim)
        rows = [
            (i, c, v)
            for i, (c, v) in enumerate(zip(chunks, vectors, strict=False))
        ]
        note_id = store.upsert_note_with_chunks(user_id, source, title, rows)
        return note_id
