from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from atsiitsiin.config import AtsiiitsiinConfig, get_snowflake_connection
from atsiitsiin.memory.store import SnowflakeMemoryStore

LOGGER = logging.getLogger("atsiitsiin.pipeline.tagging")


def _completion_call(**kwargs: Any) -> Any:
    """Deferred import for litellm to avoid mandatory dependency during type checks."""
    completion = import_module("litellm").completion
    return completion(**kwargs)


class TaggingError(Exception):
    """Raised when tag classification fails."""


@dataclass
class TagSuggestion:
    tag: str
    confidence: float | None = None
    display_name: str | None = None
    description: str | None = None
    color_hex: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "confidence": self.confidence,
            "display_name": self.display_name,
            "description": self.description,
            "color_hex": self.color_hex,
            "source": self.source,
        }


@dataclass
class TaggingResult:
    note_id: str
    tags: list[TagSuggestion]
    audit_id: str | None
    raw_response: Any


SYSTEM_PROMPT = (
    "You are a knowledge organization assistant that assigns descriptive tags to notes. "
    "Given a note title and body, return 3 to 5 concise tags (single words or short phrases) "
    "that summarize the key themes. Respond with strict JSON like:\n"
    '{"tags": [{"tag": "example", "confidence": 0.8}]}\n'
    "Include confidence between 0 and 1 when possible. Tags should be lowercase kebab-case."
)


def _extract_note_content(note: dict[str, Any]) -> str:
    title = note.get("TITLE") or ""
    description = note.get("DESCRIPTION") or ""
    return f"Title: {title}\n\nContent:\n{description}".strip()


def _strip_code_fences(content: str) -> str:
    cleaned = content.strip()
    if not cleaned.startswith("```"):
        return cleaned
    parts = cleaned.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return cleaned.strip("`")


def _suggestion_from_entry(entry: dict[str, Any], source: str) -> TagSuggestion | None:
    tag_value = entry.get("tag")
    if not isinstance(tag_value, str):
        return None
    tag = tag_value.strip()
    if not tag:
        return None
    return TagSuggestion(
        tag=tag,
        confidence=_safe_float(entry.get("confidence")),
        display_name=entry.get("display_name"),
        description=entry.get("description"),
        color_hex=entry.get("color_hex"),
        source=source,
    )


def _parse_llm_response(content: str | None, *, source: str) -> list[TagSuggestion]:
    if not content:
        raise TaggingError("LLM returned empty response")

    cleaned = _strip_code_fences(content)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TaggingError(f"Failed to parse LLM response as JSON: {exc}") from exc

    tags = payload.get("tags")
    if not isinstance(tags, Sequence):
        raise TaggingError("LLM response missing 'tags' array")

    suggestions = [
        suggestion
        for entry in tags
        if isinstance(entry, dict)
        for suggestion in [_suggestion_from_entry(entry, source)]
        if suggestion is not None
    ]

    if not suggestions:
        raise TaggingError("LLM response did not include any valid tags")

    return suggestions


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_note_tags(
    note_id: str,
    cfg: AtsiiitsiinConfig,
    *,
    max_tags: int = 5,
    source: str = "llm_suggestion",
    temperature: float | None = None,
) -> TaggingResult:
    """
    Use the configured LLM to classify tags for a note, persist them, and return the result.
    """
    with get_snowflake_connection(cfg.snowflake_env_file).session_context() as session:
        store = SnowflakeMemoryStore(session, cfg.embedding_dim)
        note = store.get_note(note_id, include_chunks=True)
        if not note:
            raise TaggingError(f"Note {note_id} not found")

        note_text = _extract_note_content(note)
        LOGGER.debug("Classifying tags for note %s", note_id)

        try:
            response: Any = _completion_call(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Generate up to {max_tags} tags for the following note. "
                            "Return strict JSON."
                        ),
                    },
                    {"role": "user", "content": note_text},
                ],
                max_tokens=min(cfg.llm_max_tokens, 512),
                temperature=temperature if temperature is not None else cfg.llm_temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            raise TaggingError(f"LLM request failed: {exc}") from exc

        message = response.choices[0].message
        content = getattr(message, "content", None)
        try:
            tags = _parse_llm_response(content, source=source)
        except TaggingError:
            LOGGER.exception("Failed to parse tagging response for note %s", note_id)
            raise

        tag_dicts = [suggestion.to_dict() for suggestion in tags]

        try:
            store.upsert_note_tags(note_id, tag_dicts, source=source)
            audit_id = store.record_tag_audit(
                note_id,
                tag_dicts,
                raw_response=_serialize_response(message),
                source=source,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to persist tags for note %s", note_id)
            raise TaggingError(f"Failed to persist tags: {exc}") from exc

        return TaggingResult(
            note_id=note_id,
            tags=tags,
            audit_id=audit_id,
            raw_response=_serialize_response(message),
        )


def _serialize_response(message: Any) -> Any:
    if message is None:
        return None
    if hasattr(message, "model_dump"):
        return message.model_dump()  # type: ignore[call-arg]
    if hasattr(message, "__dict__"):
        return message.__dict__
    return message


def assign_tags_to_notes(
    note_ids: Iterable[str],
    cfg: AtsiiitsiinConfig,
    *,
    max_tags: int = 5,
    source: str = "llm",
) -> list[TaggingResult]:
    results: list[TaggingResult] = []
    for note_id in note_ids:
        try:
            result = classify_note_tags(
                note_id,
                cfg,
                max_tags=max_tags,
                source=source,
            )
            results.append(result)
        except TaggingError as exc:
            LOGGER.error("Tagging failed for note %s: %s", note_id, exc)
    return results


