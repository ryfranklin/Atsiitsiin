# ruff: noqa: BLE001
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

from atsiitsiin.config import AtsiiitsiinConfig, get_snowflake_connection

LOGGER = logging.getLogger("atsiitsiin.pipeline.sentiment")


ContextLabel = str


CLASSIFICATION_SYSTEM_PROMPT = (
    "You analyze personal notes and decide whether they primarily describe work, personal, "
    "or leisure activities. Return a concise JSON object with `label` (one of: work, personal, "
    "leisure, mixed, unknown) and `confidence` between 0 and 1. When the note mixes multiple "
    "domains, use `mixed`. If you truly cannot decide, use `unknown`."
)


CATEGORY_ALIASES = {
    "work": "work",
    "professional": "work",
    "career": "work",
    "job": "work",
    "office": "work",
    "personal": "personal",
    "family": "personal",
    "home": "personal",
    "life": "personal",
    "wellness": "personal",
    "health": "personal",
    "leisure": "leisure",
    "hobby": "leisure",
    "travel": "leisure",
    "entertainment": "leisure",
    "recreation": "leisure",
    "mixed": "mixed",
    "other": "mixed",
    "unknown": "unknown",
}


KEYWORD_WEIGHTS = {
    "work": {
        "project": 1.0,
        "client": 1.0,
        "meeting": 0.8,
        "deadline": 1.2,
        "deliverable": 1.0,
        "proposal": 0.8,
        "launch": 0.8,
        "retro": 0.6,
        "sprint": 0.6,
        "strategy": 0.5,
        "team": 0.4,
    },
    "personal": {
        "family": 1.0,
        "doctor": 1.0,
        "appointment": 0.7,
        "kids": 0.8,
        "child": 0.8,
        "parent": 0.7,
        "finance": 0.6,
        "budget": 0.6,
        "home": 0.6,
        "errand": 0.8,
        "self-care": 0.8,
        "health": 0.9,
        "relationship": 1.0,
    },
    "leisure": {
        "vacation": 1.2,
        "trip": 0.8,
        "travel": 1.2,
        "hike": 0.9,
        "game": 0.6,
        "gaming": 0.7,
        "movie": 0.8,
        "concert": 1.0,
        "book": 0.5,
        "dinner": 0.5,
        "restaurant": 0.6,
        "festival": 1.0,
        "weekend": 0.7,
        "hobby": 0.9,
    },
}


class SentimentProcessingError(Exception):
    """Raised when a note fails sentiment processing."""

    def __init__(self, note_id: str, message: str):
        super().__init__(message)
        self.note_id = note_id


@dataclass
class SentimentJobLogEntry:
    note_id: str
    status: str
    message: str | None = None
    label: str | None = None
    score: float | None = None


@dataclass
class SentimentJobResult:
    processed: int
    skipped: int
    failed: int
    logs: list[SentimentJobLogEntry]
    work_notes: int = 0
    personal_notes: int = 0
    leisure_notes: int = 0
    mixed_notes: int = 0
    unknown_notes: int = 0
    average_context_confidence: float | None = None


def run_sentiment_job(
    limit: int = 100,
    cfg: AtsiiitsiinConfig | None = None,
    logger: logging.Logger | None = None,
    on_entry: Callable[[SentimentJobLogEntry], None] | None = None,
) -> SentimentJobResult:
    """Run sentiment analysis for pending notes and return a summary."""
    config = cfg or AtsiiitsiinConfig()
    log = logger or LOGGER
    logs: list[SentimentJobLogEntry] = []
    processed = skipped = failed = 0
    work_notes = personal_notes = leisure_notes = mixed_notes = unknown_notes = 0
    confidence_total = 0.0
    confidence_counter = 0

    connection = get_snowflake_connection(config.snowflake_env_file)
    with connection.session_context() as session:
        ensure_schema(session)
        for note in fetch_pending_notes(session, limit=limit):
            note_id = note["NOTE_ID"]
            title = note.get("TITLE") or ""
            if not title:
                log_work(session, note_id, "SKIPPED", error="Note has no text")
                skipped_entry = SentimentJobLogEntry(
                    note_id=note_id,
                    status="SKIPPED",
                    message="Note has no text",
                )
                logs.append(skipped_entry)
                _emit(on_entry, skipped_entry)
                skipped += 1
                continue

            try:
                label, score = _process_note(
                    session,
                    note_id,
                    title,
                    config,
                )
                normalized_label = _normalize_label(label)
                if normalized_label == "work":
                    work_notes += 1
                elif normalized_label == "personal":
                    personal_notes += 1
                elif normalized_label == "leisure":
                    leisure_notes += 1
                elif normalized_label == "mixed":
                    mixed_notes += 1
                else:
                    unknown_notes += 1
                if score is not None:
                    confidence_total += score
                    confidence_counter += 1
                log.info(
                    "Processed sentiment for note %s: %s %.3f",
                    note_id,
                    label,
                    score,
                )
                success_entry = SentimentJobLogEntry(
                    note_id=note_id,
                    status="SUCCESS",
                    label=label,
                    score=score,
                )
                logs.append(success_entry)
                _emit(on_entry, success_entry)
                processed += 1
            except SentimentProcessingError as exc:
                log.exception("Failed processing note %s", exc.note_id)
                log_work(session, exc.note_id, "FAILED", error=str(exc))
                failed_entry = SentimentJobLogEntry(
                    note_id=exc.note_id,
                    status="FAILED",
                    message=str(exc),
                )
                logs.append(failed_entry)
                _emit(on_entry, failed_entry)
                failed += 1

    return SentimentJobResult(
        processed=processed,
        skipped=skipped,
        failed=failed,
        logs=logs,
        work_notes=work_notes,
        personal_notes=personal_notes,
        leisure_notes=leisure_notes,
        mixed_notes=mixed_notes,
        unknown_notes=unknown_notes,
        average_context_confidence=(
            confidence_total / confidence_counter if confidence_counter else None
        ),
    )


def ensure_schema(session: Session) -> None:
    session.sql(
        "ALTER TABLE IF EXISTS NOTES ADD COLUMN IF NOT EXISTS CONTEXT_LABEL STRING"
    ).collect()
    session.sql(
        "ALTER TABLE IF EXISTS NOTES ADD COLUMN IF NOT EXISTS CONTEXT_CONFIDENCE FLOAT"
    ).collect()
    session.sql(
        "ALTER TABLE IF EXISTS NOTES ADD COLUMN IF NOT EXISTS CONTEXT_ANALYZED_AT TIMESTAMP_NTZ"
    ).collect()
    session.sql(
        """
        CREATE TABLE IF NOT EXISTS NOTE_CONTEXT_WORKLOG (
            WORKLOG_ID STRING DEFAULT UUID_STRING(),
            NOTE_ID STRING,
            STATUS STRING,
            CONTEXT_LABEL STRING,
            CONTEXT_CONFIDENCE FLOAT,
            ERROR_MESSAGE STRING,
            ANALYZED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (WORKLOG_ID)
        )
        """
    ).collect()


def fetch_pending_notes(session: Session, limit: int = 100) -> list[dict[str, Any]]:
    query = (
        session.table("NOTES")
        .select(
            col("ID").alias("NOTE_ID"),
            col("TITLE"),
            col("CONTEXT_ANALYZED_AT"),
        )
        .where(col("CONTEXT_ANALYZED_AT").isNull())
        .limit(limit)
    )
    return [row.as_dict() for row in query.collect()]


def update_note_context(
    session: Session,
    note_id: str,
    label: str,
    score: float,
) -> None:
    session.sql(
        """
        UPDATE NOTES
        SET CONTEXT_LABEL = ?, CONTEXT_CONFIDENCE = ?, CONTEXT_ANALYZED_AT = CURRENT_TIMESTAMP()
        WHERE ID = ?
        """,
        params=[label, score, note_id],
    ).collect()


def log_work(
    session: Session,
    note_id: str,
    status: str,
    label: str | None = None,
    score: float | None = None,
    error: str | None = None,
) -> None:
    session.sql(
        """
        INSERT INTO NOTE_CONTEXT_WORKLOG
            (NOTE_ID, STATUS, CONTEXT_LABEL, CONTEXT_CONFIDENCE, ERROR_MESSAGE)
        VALUES (?, ?, ?, ?, ?)
        """,
        params=[note_id, status, label, score, error],
    ).collect()


def _process_note(
    session: Session,
    note_id: str,
    title: str,
    cfg: AtsiiitsiinConfig,
) -> tuple[str, float]:
    try:
        label, score = classify_note_context(session, note_id, title, cfg)
        update_note_context(session, note_id, label, score)
        log_work(session, note_id, "SUCCESS", label=label, score=score)
        return label, score
    except Exception as exc:  # noqa: B902
        raise SentimentProcessingError(note_id, str(exc)) from exc


def _emit(
    callback: Callable[[SentimentJobLogEntry], None] | None,
    entry: SentimentJobLogEntry,
) -> None:
    if callback:
        callback(entry)


def classify_note_context(
    session: Session,
    note_id: str,
    title: str,
    cfg: AtsiiitsiinConfig,
) -> tuple[str, float]:
    """
    Determine whether a note is primarily work, personal, or leisure focused.
    Falls back to heuristic scoring if the LLM is unavailable.
    """
    note_body = _load_note_body(session, note_id)
    tag_values = _load_note_tags(session, note_id)
    combined_text = "\n\n".join(
        part for part in [title, note_body] if isinstance(part, str) and part.strip()
    )

    try:
        label, confidence = _classify_with_llm(
            cfg,
            combined_text,
            tag_values,
        )
        if label:
            return label, confidence
    except (RuntimeError, ValueError) as exc:
        LOGGER.exception(
            "LLM context classification failed for note %s: %s", note_id, exc
        )

    label, confidence = _heuristic_classification(combined_text, tag_values)
    return label, confidence


def _completion_call(**kwargs: Any) -> Any:
    completion = import_module("litellm").completion
    return completion(**kwargs)


def _classify_with_llm(
    cfg: AtsiiitsiinConfig,
    text: str,
    tags: list[str],
) -> tuple[ContextLabel, float]:
    if not cfg.llm_model:
        raise ValueError("LLM model not configured")
    prompt = (
        "Classify the following journal-style note. "
        "Respond with JSON: {\"label\": \"work|personal|leisure|mixed|unknown\", \"confidence\": 0-1, \"explanation\": \"...\"}."
        "\n\n"
        f"Tags: {', '.join(tags) if tags else 'none'}\n\n"
        f"Note:\n{text}"
    )
    try:
        response: Any = _completion_call(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=min(cfg.llm_max_tokens, 512),
            temperature=max(min(cfg.llm_temperature, 0.8), 0.0),
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("LLM classification returned empty content")
    payload = _parse_json_payload(content)
    label_raw = str(payload.get("label", "")).strip().lower()
    label = CATEGORY_ALIASES.get(label_raw, label_raw or "unknown")
    confidence_raw = payload.get("confidence") or payload.get("score")
    confidence = _safe_float(confidence_raw)
    if confidence is None:
        confidence = 0.65
    confidence = max(0.0, min(float(confidence), 1.0))
    return label, confidence


def _parse_json_payload(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        segments = cleaned.split("```")
        if len(segments) >= 3:
            cleaned = segments[1].strip()
        else:
            cleaned = cleaned.strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM JSON response: {exc}") from exc


def _heuristic_classification(
    text: str,
    tags: list[str],
) -> tuple[ContextLabel, float]:
    lowered = text.lower()
    tag_lower = [tag.lower() for tag in tags]
    scores = {"work": 0.1, "personal": 0.1, "leisure": 0.1}

    for category, keywords in KEYWORD_WEIGHTS.items():
        for keyword, weight in keywords.items():
            if keyword in lowered:
                scores[category] += weight
    for tag in tag_lower:
        for alias, mapped in CATEGORY_ALIASES.items():
            if alias in tag and mapped in scores:
                scores[mapped] += 0.75

    top_category = max(scores, key=lambda category: scores[category])
    total_score = sum(scores.values())
    confidence = scores[top_category] / total_score if total_score else 0.33
    return top_category, float(confidence)


def _normalize_label(label: str | None) -> str:
    if not label:
        return "unknown"
    label_lower = label.lower().strip()
    return CATEGORY_ALIASES.get(label_lower, label_lower or "unknown")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_note_body(session: Session, note_id: str) -> str:
    rows = session.sql(
        """
        SELECT TEXT
        FROM NOTE_CHUNKS
        WHERE NOTE_ID = ?
        ORDER BY CHUNK_INDEX
        """,
        params=[note_id],
    ).collect()
    parts: list[str] = []
    for row in rows:
        value: Any
        if hasattr(row, "as_dict"):
            value = row.as_dict().get("TEXT")
        elif isinstance(row, dict):
            value = row.get("TEXT")
        else:
            value = None
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n\n".join(parts)


def _load_note_tags(session: Session, note_id: str) -> list[str]:
    rows = session.sql(
        """
        SELECT TAG
        FROM NOTE_TAGS
        WHERE NOTE_ID = ?
        """,
        params=[note_id],
    ).collect()
    tags: list[str] = []
    for row in rows:
        value: Any
        if hasattr(row, "as_dict"):
            value = row.as_dict().get("TAG")
        elif isinstance(row, dict):
            value = row.get("TAG")
        else:
            value = None
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                tags.append(trimmed)
    return tags

