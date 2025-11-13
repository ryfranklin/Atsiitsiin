from __future__ import annotations

from typing import Any

from starlette.concurrency import run_in_threadpool

from atsiitsiin.config import AtsiiitsiinConfig, get_snowflake_connection
from atsiitsiin.memory import (
    count_notes,
    ingest_note,
    list_notes,
    search_memory,
    sentiment_summary,
)
from atsiitsiin.memory import get_note as get_note_from_store
from atsiitsiin.memory.store import SnowflakeMemoryStore
from atsiitsiin.pipelines.tagging import assign_tags_to_notes

from ..schemas.notes import NoteCreateRequest, NoteTagInput


async def create_note(
    payload: NoteCreateRequest, config: AtsiiitsiinConfig
) -> str:
    def _ingest() -> str:
        note_id = ingest_note(
            user_id=payload.user_id,
            source=payload.source,
            title=payload.title,
            content=payload.content,
            cfg=config,
        )
        if payload.tags:
            tag_dicts = [
                {
                    "tag": tag.tag,
                    "display_name": tag.display_name or tag.tag,
                    "description": tag.description,
                    "color_hex": tag.color_hex,
                    "confidence": tag.confidence,
                    "source": "user",
                }
                for tag in payload.tags
            ]
            with get_snowflake_connection(
                config.snowflake_env_file
            ).session_context() as session:
                store = SnowflakeMemoryStore(session, config.embedding_dim)
                store.upsert_note_tags(note_id, tag_dicts, source="user")
        return note_id

    return await run_in_threadpool(_ingest)


async def search_notes(
    query: str, config: AtsiiitsiinConfig, limit: int
) -> list[dict[str, Any]]:
    def _search() -> list[dict[str, Any]]:
        return search_memory(query=query, cfg=config, k=limit)

    return await run_in_threadpool(_search)


async def fetch_notes(
    config: AtsiiitsiinConfig, limit: int, offset: int
) -> tuple[list[dict[str, Any]], int]:
    def _list() -> list[dict[str, Any]]:
        return list_notes(cfg=config, limit=limit, offset=offset)

    def _count() -> int:
        return count_notes(cfg=config)

    notes_task = run_in_threadpool(_list)
    total_task = run_in_threadpool(_count)
    notes = await notes_task
    total = await total_task
    return notes, total


async def fetch_note(
    note_id: str, config: AtsiiitsiinConfig, include_chunks: bool
) -> dict[str, Any] | None:
    def _get() -> dict[str, Any] | None:
        return get_note_from_store(
            note_id=note_id, cfg=config, include_chunks=include_chunks
        )

    return await run_in_threadpool(_get)


async def get_sentiment_stats(config: AtsiiitsiinConfig) -> dict[str, Any]:
    def _summary() -> dict[str, Any]:
        return sentiment_summary(cfg=config)

    return await run_in_threadpool(_summary)


async def run_tagging(
    config: AtsiiitsiinConfig,
    *,
    limit: int = 25,
    note_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    def _tag() -> tuple[list[dict[str, Any]], list[str]]:
        target_ids = list(note_ids or [])

        if not target_ids:
            with get_snowflake_connection(
                config.snowflake_env_file
            ).session_context() as session:
                store = SnowflakeMemoryStore(session, config.embedding_dim)
                target_ids = store.list_note_ids_missing_tags(limit)

        if not target_ids:
            return [], []

        results = assign_tags_to_notes(target_ids, config, source="llm_suggestion")
        tagged_payloads: list[dict[str, Any]] = []
        for result in results:
            tagged_payloads.append(
                {
                    "note_id": result.note_id,
                    "tags": [suggestion.to_dict() for suggestion in result.tags],
                    "audit_id": result.audit_id,
                }
            )
        tagged_ids = {item["note_id"] for item in tagged_payloads}
        skipped = [note_id for note_id in target_ids if note_id not in tagged_ids]
        return tagged_payloads, skipped

    return await run_in_threadpool(_tag)


async def apply_user_tags(
    note_id: str,
    tags: list[NoteTagInput],
    config: AtsiiitsiinConfig,
) -> list[dict[str, Any]]:
    def _apply() -> list[dict[str, Any]]:
        tag_payloads = [
            {
                "tag": tag.tag,
                "display_name": tag.display_name or tag.tag,
                "description": tag.description,
                "color_hex": tag.color_hex,
                "confidence": tag.confidence,
                "source": "user",
            }
            for tag in tags
        ]
        with get_snowflake_connection(
            config.snowflake_env_file
        ).session_context() as session:
            store = SnowflakeMemoryStore(session, config.embedding_dim)
            note = store.get_note(note_id, include_chunks=False)
            if note is None:
                raise KeyError(note_id)
            if tag_payloads:
                store.upsert_note_tags(note_id, tag_payloads, source="user")
            return store.list_note_tags(note_id)

    try:
        return await run_in_threadpool(_apply)
    except KeyError as exc:
        raise ValueError(f"Note {exc.args[0]} not found") from exc


async def delete_note(
    note_id: str,
    config: AtsiiitsiinConfig,
) -> None:
    def _delete() -> None:
        with get_snowflake_connection(
            config.snowflake_env_file
        ).session_context() as session:
            store = SnowflakeMemoryStore(session, config.embedding_dim)
            deleted = store.delete_note(note_id)
            if not deleted:
                raise KeyError(note_id)

    try:
        await run_in_threadpool(_delete)
    except KeyError as exc:
        raise ValueError(f"Note {exc.args[0]} not found") from exc


async def remove_note_tags(
    note_id: str,
    tag_values: list[str],
    config: AtsiiitsiinConfig,
) -> list[dict[str, Any]]:
    def _remove() -> list[dict[str, Any]]:
        with get_snowflake_connection(
            config.snowflake_env_file
        ).session_context() as session:
            store = SnowflakeMemoryStore(session, config.embedding_dim)
            note = store.get_note(note_id, include_chunks=False)
            if note is None:
                raise KeyError(note_id)
            return store.remove_note_tags(note_id, tag_values)

    try:
        return await run_in_threadpool(_remove)
    except KeyError as exc:
        raise ValueError(f"Note {exc.args[0]} not found") from exc

