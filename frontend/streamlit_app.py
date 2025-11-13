"""Streamlit prototype UI for Atsiitsʼiin."""

from __future__ import annotations

import html
import logging
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import requests
import streamlit as st

LOG_LEVEL_NAME = os.getenv("ATSIIITSIN_STREAMLIT_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
else:
    root_logger.setLevel(LOG_LEVEL)

LOGGER = logging.getLogger("atsiitsiin.streamlit")
LOGGER.setLevel(LOG_LEVEL)

API_BASE_URL = os.getenv("ATSIIITSIN_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("ATSIIITSIN_API_TIMEOUT", "60"))


def _normalize_dict(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {str(k).lower(): v for k, v in record.items()}
    if "note_id" not in normalized and "id" in normalized:
        normalized["note_id"] = normalized["id"]
    if "user_id" not in normalized and "user" in normalized:
        normalized["user_id"] = normalized["user"]
    if "text" not in normalized and "content" in normalized:
        normalized["text"] = normalized["content"]
    return normalized


def _normalize_note(record: dict[str, Any]) -> dict[str, Any]:
    note = _normalize_dict(record)
    note.setdefault("title", record.get("TITLE") or record.get("title") or "Untitled")
    note.setdefault("source", record.get("SOURCE") or record.get("source") or "unknown")
    tags = record.get("tags") or record.get("TAGS")
    if isinstance(tags, list):
        note["tags"] = [
            _normalize_dict(tag) for tag in tags if isinstance(tag, dict)
        ]
    else:
        note["tags"] = []
    return note


def _normalize_chunk(record: dict[str, Any]) -> dict[str, Any]:
    chunk = _normalize_dict(record)
    chunk.setdefault("text", record.get("TEXT") or chunk.get("text", ""))
    chunk.setdefault("chunk_index", record.get("CHUNK_INDEX") or chunk.get("chunk_index", 0))
    chunk.setdefault("note_id", record.get("NOTE_ID") or chunk.get("note_id"))
    chunk.setdefault("chunk_id", record.get("ID") or chunk.get("chunk_id"))
    return chunk


def _normalize_search_hit(record: dict[str, Any]) -> dict[str, Any]:
    hit = _normalize_dict(record)
    hit.setdefault("note_id", record.get("NOTE_ID") or hit.get("note_id"))
    hit.setdefault("chunk_id", record.get("ID") or hit.get("chunk_id"))
    hit.setdefault("chunk_index", record.get("CHUNK_INDEX") or hit.get("chunk_index", 0))
    hit.setdefault("text", record.get("TEXT") or hit.get("text", ""))
    hit.setdefault("distance", record.get("distance", 0.0))
    return hit


def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        LOGGER.info(
            "GET request url=%s params=%s",
            f"{API_BASE_URL}{path}",
            params or {},
        )
        response = requests.get(
            f"{API_BASE_URL}{path}",
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        LOGGER.info(
            "GET response url=%s status=%s content=%s",
            f"{API_BASE_URL}{path}",
            response.status_code,
            response.text,
        )
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover - UI feedback
        LOGGER.error(
            "GET request failed url=%s params=%s error=%s",
            f"{API_BASE_URL}{path}",
            params or {},
            exc,
        )
        st.error(f"Request failed: {exc}")
        return None


def _api_post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        LOGGER.info("POST request url=%s payload=%s", f"{API_BASE_URL}{path}", payload)
        response = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        LOGGER.info(
            "POST response url=%s status=%s content=%s",
            f"{API_BASE_URL}{path}",
            response.status_code,
            response.text,
        )
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover - UI feedback
        LOGGER.error(
            "POST request failed url=%s payload=%s error=%s",
            f"{API_BASE_URL}{path}",
            payload,
            exc,
        )
        st.error(f"Request failed: {exc}")
        return None


def _api_delete(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        LOGGER.info("DELETE request url=%s payload=%s", f"{API_BASE_URL}{path}", payload)
        response = requests.delete(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        if response.text:
            LOGGER.info(
                "DELETE response url=%s status=%s content=%s",
                f"{API_BASE_URL}{path}",
                response.status_code,
                response.text,
            )
            return response.json()
        LOGGER.info(
            "DELETE response url=%s status=%s",
            f"{API_BASE_URL}{path}",
            response.status_code,
        )
        return None
    except requests.RequestException as exc:  # pragma: no cover - UI feedback
        LOGGER.error(
            "DELETE request failed url=%s payload=%s error=%s",
            f"{API_BASE_URL}{path}",
            payload,
            exc,
        )
        st.error(f"Request failed: {exc}")
        return None


def run_tagging_job(limit: int, note_ids: list[str] | None = None) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"limit": limit}
    if note_ids:
        payload["note_ids"] = note_ids
    return _api_post("/notes/analytics/tagging/run", payload)


def get_config_snapshot() -> dict[str, Any] | None:
    data = _api_get("/config")
    if isinstance(data, dict):
        return data
    return None


def get_notes(limit: int = 20, offset: int = 0) -> dict[str, Any]:
    data = _api_get("/notes", params={"limit": limit, "offset": offset}) or {}
    notes_raw = data.get("notes", [])
    notes_normalized = [
        _normalize_note(note) for note in notes_raw if isinstance(note, dict)
    ]
    return {
        "notes": notes_normalized,
        "total": data.get("total", 0),
    }


@st.cache_data(show_spinner=False)
def get_note_detail(note_id: str, include_chunks: bool = True) -> dict[str, Any] | None:
    raw_note = _api_get(
        f"/notes/{note_id}",
        params={"include_chunks": "true" if include_chunks else "false"},
    )
    if not isinstance(raw_note, dict):
        st.cache_data.clear()
        return None
    note = _normalize_note(raw_note)
    if include_chunks:
        raw_chunks = raw_note.get("chunks", [])
        note["chunks"] = [
            _normalize_chunk(chunk)
            for chunk in raw_chunks
            if isinstance(chunk, dict)
        ]
    return note


@st.cache_data(show_spinner=False)
def search_notes(query: str, limit: int = 8) -> dict[str, Any]:
    data = _api_get("/notes/search", params={"q": query, "k": limit}) or {}
    raw_hits = data.get("results", [])
    normalized_hits = [
        _normalize_search_hit(hit) for hit in raw_hits if isinstance(hit, dict)
    ]
    return {
        "query": data.get("query", query),
        "results": normalized_hits,
    }


def create_note(
    title: str,
    content: str,
    user_id: str,
    source: str,
    *,
    tags: list[dict[str, Any]] | None = None,
) -> str | None:
    st.cache_data.clear()
    payload: dict[str, Any] = {
        "title": title,
        "content": content,
        "user_id": user_id,
        "source": source,
    }
    if tags:
        payload["tags"] = tags
    LOGGER.info(
        "Submitting note create request",
        extra={"api_base": API_BASE_URL, "payload": payload},
    )
    data = _api_post("/notes", payload)
    if data and "note_id" in data:
        LOGGER.info(
            "Note created successfully",
            extra={"note_id": data["note_id"], "api_base": API_BASE_URL},
        )
        return data["note_id"]
    LOGGER.warning(
        "Note creation failed or returned unexpected payload",
        extra={"response": data, "api_base": API_BASE_URL},
    )
    return None


def add_user_tags_to_note(
    note_id: str, tags: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not tags:
        return None
    payload = {"tags": tags}
    return _api_post(f"/notes/{note_id}/tags", payload)


def remove_tags_from_note(note_id: str, tags: list[str]) -> dict[str, Any] | None:
    if not tags:
        return None
    payload = {"tags": tags}
    return _api_delete(f"/notes/{note_id}/tags", payload)


def delete_note_api(note_id: str) -> dict[str, Any] | None:
    return _api_delete(f"/notes/{note_id}")


def get_sentiment_stats() -> dict[str, Any]:
    return _api_get("/notes/analytics/sentiment") or {}


def run_sentiment_job(limit: int) -> dict[str, Any] | None:
    payload = {"limit": limit}
    return _api_post("/notes/analytics/sentiment/run", payload)


def list_sentiment_jobs() -> list[dict[str, Any]]:
    data = _api_get("/notes/analytics/sentiment/jobs") or {}
    jobs = data.get("jobs", [])
    if isinstance(jobs, list):
        processed: list[dict[str, Any]] = []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            normalized = _normalize_dict(job)
            summary = normalized.get("summary")
            if isinstance(summary, dict):
                normalized["summary"] = _normalize_dict(summary)
            processed.append(normalized)
        return processed
    return []


def get_sentiment_job(job_id: str) -> dict[str, Any] | None:
    job = _api_get(f"/notes/analytics/sentiment/jobs/{job_id}")
    if isinstance(job, dict):
        normalized = _normalize_dict(job)
        summary = normalized.get("summary")
        if isinstance(summary, dict):
            normalized["summary"] = _normalize_dict(summary)
        logs = normalized.get("logs")
        if isinstance(logs, list):
            normalized["logs"] = [
                _normalize_dict(entry) for entry in logs if isinstance(entry, dict)
            ]
        return normalized
    return None


def format_timestamp(value: str | None) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.strftime("%b %d, %Y · %I:%M %p")


def _render_tag_pills(tags: list[dict[str, Any]]) -> str:
    if not tags:
        return ""
    parts: list[str] = []
    for tag in tags:
        tag_value = tag.get("TAG") or tag.get("tag")
        if not isinstance(tag_value, str):
            continue
        display = tag.get("DISPLAY_NAME") or tag_value
        source_raw = tag.get("SOURCE") or tag.get("source")
        source = str(source_raw).lower() if isinstance(source_raw, str) else ""
        color = tag.get("COLOR_HEX")
        if not isinstance(color, str) or not color.strip():
            color = "#1a7f37" if source == "user" else "#0070c9"
        pill_title_raw = display if source == "" else f"{display} · {source}"
        pill_title = html.escape(pill_title_raw)
        pill_text = html.escape(display)
        parts.append(
            f"<span style='display:inline-block;background:{color};color:white;padding:2px 10px;border-radius:12px;margin-right:6px;margin-bottom:4px;font-size:0.75rem;' title='{pill_title}'>"
            f"{pill_text}"
            "</span>"
        )
    if not parts:
        return ""
    return "<div style='margin-top:4px;margin-bottom:6px;'>" + "".join(parts) + "</div>"


def _filter_notes_by_tags(
    notes: list[dict[str, Any]], selected_tags: list[str]
) -> list[dict[str, Any]]:
    if not selected_tags:
        return notes
    selected = {tag for tag in selected_tags if isinstance(tag, str)}
    if not selected:
        return notes
    filtered: list[dict[str, Any]] = []
    for note in notes:
        note_tag_values = {
            tag_value
            for tag in note.get("tags", [])
            if isinstance(tag, dict)
            for tag_value in [tag.get("TAG") or tag.get("tag")]
            if isinstance(tag_value, str) and tag_value.strip()
        }
        if note_tag_values & selected:
            filtered.append(note)
    return filtered


def _render_note_tags_add_form(note_id: str, reset_key: str) -> None:
    input_key = f"note-tags-input-{note_id}"
    st.session_state.setdefault(input_key, "")
    if st.session_state.pop(reset_key, False):
        st.session_state[input_key] = ""

    new_tags_raw = st.text_input(
        "Add tags to this note",
        key=input_key,
        placeholder="project-phoenix, urgent-review",
        help="Provide comma or newline separated tags to add. We'll keep your tags alongside suggestions.",
    )
    parsed_note_tags = _parse_user_tag_input(new_tags_raw)
    if parsed_note_tags:
        preview_payload = [
            {
                "tag": tag["tag"],
                "display_name": tag["display_name"],
                "source": "user",
            }
            for tag in parsed_note_tags
        ]
        st.markdown(
            _render_tag_pills(preview_payload),
            unsafe_allow_html=True,
        )
    add_button_key = f"note-tags-button-{note_id}"
    if st.button("Save tags", key=add_button_key):
        if not parsed_note_tags:
            st.warning("Enter at least one tag before saving.")
            return
        with st.spinner("Applying tags..."):
            response = add_user_tags_to_note(note_id, parsed_note_tags)
        if response and response.get("tags") is not None:
            st.success("Tags saved to note.")
            st.cache_data.clear()
            st.session_state[reset_key] = True
            st.rerun()
        else:
            st.error("Unable to save tags. Please try again.")

def _build_tag_remove_options(
    tags: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    values: list[str] = []
    label_map: dict[str, str] = {}
    for tag in tags:
        tag_value = tag.get("TAG") or tag.get("tag")
        if not isinstance(tag_value, str):
            continue
        normalized = tag_value.strip()
        if not normalized:
            continue
        display = tag.get("DISPLAY_NAME") or normalized
        source = (tag.get("SOURCE") or "").lower()
        label = f"{display} · {source}" if source else display
        values.append(normalized)
        label_map[normalized] = label
    return values, label_map


def _render_note_tags_remove_form(
    note_id: str,
    current_tags: list[dict[str, Any]],
    reset_key: str,
) -> None:
    values, label_map = _build_tag_remove_options(current_tags)
    if not values:
        return

    remove_key = f"note-tags-remove-select-{note_id}"

    def _format_label(value: str) -> str:
        return label_map.get(value, value)

    selected_remove = st.multiselect(
        "Remove tags from this note",
        options=values,
        format_func=_format_label,
        key=remove_key,
    )
    if st.button("Remove selected tags", key=f"note-tags-remove-btn-{note_id}"):
        if not selected_remove:
            st.warning("Select at least one tag to remove.")
            return
        with st.spinner("Removing tags..."):
            response = remove_tags_from_note(note_id, selected_remove)
        if response and response.get("tags") is not None:
            st.success("Tags removed.")
            st.cache_data.clear()
            st.session_state[reset_key] = True
            st.rerun()
        else:
            st.error("Unable to remove tags. Please try again.")


def _render_note_tags_editor(
    note_id: str, current_tags: list[dict[str, Any]] | None
) -> None:
    reset_key = f"note-tags-reset-{note_id}"
    _render_note_tags_add_form(note_id, reset_key)
    if current_tags:
        _render_note_tags_remove_form(note_id, current_tags, reset_key)


def _render_note_panel(note: dict[str, Any]) -> None:
    note_title = note.get("title", "Untitled")
    with st.expander(note_title, expanded=False):
        _render_note_metadata(note)
        if isinstance(note.get("tags"), list):
            _render_existing_tags(note["tags"])
        note_id = note.get("note_id")
        detail = get_note_detail(note_id or "")
        combined_tags = _collect_note_tags(note, detail)
        if note_id:
            _render_note_tags_editor(note_id, combined_tags)
            _render_note_delete_controls(note_id, note_title)
        _render_note_chunks(detail)


def _render_note_metadata(note: dict[str, Any]) -> None:
    created = format_timestamp(note.get("created_at"))
    source = note.get("source", "unknown")
    user_id = note.get("user_id", "unknown")
    context_label = note.get("context_label") or note.get("sentiment_label")
    confidence = note.get("context_confidence")
    if confidence is None:
        confidence = note.get("sentiment_score")
    context_summary = _format_context_summary(context_label, confidence)
    classified_at = note.get("context_analyzed_at") or note.get("sentiment_analyzed_at")
    parts = [
        created,
        f"Source: {source}",
        f"User: {user_id}",
        f"Context: {context_summary}",
    ]
    if classified_at:
        parts.append(f"Classified: {format_timestamp(classified_at)}")
    st.caption(" · ".join(parts))


def _render_existing_tags(tags: list[dict[str, Any]]) -> None:
    if not tags:
        return
    st.markdown(
        _render_tag_pills(tags),
        unsafe_allow_html=True,
    )


def _collect_note_tags(
    note: dict[str, Any],
    detail: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if detail and isinstance(detail.get("tags"), list):
        return [tag for tag in detail["tags"] if isinstance(tag, dict)]
    if isinstance(note.get("tags"), list):
        return [tag for tag in note["tags"] if isinstance(tag, dict)]
    return []


def _render_note_delete_controls(note_id: str, note_title: str) -> None:
    confirm_key = f"delete-note-confirm-{note_id}"
    delete_confirm = st.checkbox(
        "Confirm delete",
        key=confirm_key,
        help="Check before deleting this note.",
    )
    if st.button("Delete note", key=f"delete-note-btn-{note_id}"):
        if not delete_confirm:
            st.warning("Check 'Confirm delete' before deleting this note.")
            return
        with st.spinner("Deleting note..."):
            response = delete_note_api(note_id)
        if response is None or response.get("deleted"):
            st.success("Note deleted.")
            st.cache_data.clear()
            activity_log = st.session_state.setdefault("recent_activity", [])
            activity_log.insert(
                0,
                {
                    "type": "note_deleted",
                    "title": note_title,
                    "note_id": note_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            st.rerun()
        else:
            st.error("Unable to delete note. Please try again.")


def _render_note_chunks(detail: dict[str, Any] | None) -> None:
    if not detail or not detail.get("chunks"):
        st.write("_No chunks retrieved_")
        return
    if detail.get("tags"):
        st.markdown(
            _render_tag_pills(detail["tags"]),
            unsafe_allow_html=True,
        )
    context_label = detail.get("context_label") or detail.get("sentiment_label")
    confidence = detail.get("context_confidence")
    if confidence is None:
        confidence = detail.get("sentiment_score")
    context_summary = _format_context_summary(context_label, confidence)
    analyzed_at = detail.get("context_analyzed_at") or detail.get("sentiment_analyzed_at")
    caption = f"Context: {context_summary}"
    if analyzed_at:
        caption += f" · Classified {format_timestamp(analyzed_at)}"
    st.caption(caption)
    for chunk in detail["chunks"]:
        st.write(chunk.get("text", ""))


def _parse_user_tag_input(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    entries = re.split(r"[,\\n]", raw)
    tags: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, str):
            continue
        display = entry.strip()
        if not display:
            continue
        slug = "-".join(display.lower().split())
        slug = re.sub(r"[^a-z0-9-_]", "", slug)
        if not slug:
            continue
        if slug in seen:
            continue
        seen.add(slug)
        tags.append(
            {
                "tag": slug,
                "display_name": display,
            }
        )
    return tags


def render_dashboard() -> None:
    st.subheader("Quick Overview")
    cols = st.columns(4)
    config = get_config_snapshot()
    notes_summary = get_notes(limit=5, offset=0)
    total_notes = notes_summary["total"]
    llm_model = config.get("llm_model") if config else "unknown"
    embedding_model = config.get("embedding_model") if config else "unknown"
    context_stats = get_sentiment_stats()
    pending_context = int(_get_stat_value(context_stats or {}, "pending_notes", 0) or 0)
    classified_context = max(total_notes - pending_context, 0)
    work_context = int(_get_stat_value(context_stats or {}, "work_notes", 0) or 0)

    cols[0].metric("Notes in Memory", f"{total_notes:,}")
    cols[1].metric("LLM Model", llm_model)
    cols[2].metric("Embedding Model", embedding_model)
    cols[3].metric(
        "Context Classified",
        f"{classified_context:,}",
        help=f"{pending_context:,} notes awaiting context tagging • {work_context:,} work-focused notes",
    )

    classified_ratio = 0.0
    with st.container():
        if total_notes > 0:
            classified_ratio = max(
                0.0,
                min(classified_context / total_notes, 1.0),
            )
            st.progress(
                classified_ratio,
                text=(
                    f"{classified_context:,} of {total_notes:,} notes "
                    "currently include context classification."
                ),
            )
        else:
            st.info("No notes captured yet. Create your first memory to begin.")

    context_breakdown = [
        ("Work", work_context),
        ("Personal", int(_get_stat_value(context_stats or {}, "personal_notes", 0) or 0)),
        ("Leisure", int(_get_stat_value(context_stats or {}, "leisure_notes", 0) or 0)),
        ("Mixed", int(_get_stat_value(context_stats or {}, "mixed_notes", 0) or 0)),
        ("Unknown", int(_get_stat_value(context_stats or {}, "unknown_notes", 0) or 0)),
        ("Pending", pending_context),
    ]
    if context_breakdown and any(count for _, count in context_breakdown):
        summary_rows: list[dict[str, Any]] = []
        total_for_share = max(total_notes, 1)
        for label, count in context_breakdown:
            share = (count / total_for_share) * 100
            summary_rows.append(
                {
                    "Context": label,
                    "Notes": count,
                    "Share": f"{share:.1f}%",
                }
            )
        st.markdown("#### Context Distribution")
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    st.markdown("### Recently Captured")
    if not notes_summary["notes"]:
        st.info("No notes captured yet. Create your first memory to begin.")
        return

    for note in notes_summary["notes"]:
        with st.container():
            context_summary = _format_context_summary(
                note.get("context_label") or note.get("sentiment_label"),
                note.get("context_confidence")
                if note.get("context_confidence") is not None
                else note.get("sentiment_score"),
            )
            st.markdown(
                f"**{note.get('title', 'Untitled')}**  \n"
                f"<small style='color: var(--text-color-muted);'>"
                f"{format_timestamp(note.get('created_at'))} · Source: {note.get('source', 'unknown')} · "
                f"Context: {html.escape(context_summary)}</small>",
                unsafe_allow_html=True,
            )
            if note.get("tags"):
                st.markdown(
                    _render_tag_pills(note["tags"]),
                    unsafe_allow_html=True,
                )
            note_id = note.get("note_id")
            if note_id:
                recent_notes = st.session_state.setdefault("recent_notes", {})
                if note_id not in recent_notes:
                    recent_notes[note_id] = note
            st.divider()


def render_notes_list() -> None:
    st.subheader("Browse Notes")
    limit = st.sidebar.slider(
        "Results per page", min_value=5, max_value=50, value=15, key="notes-page-size"
    )
    page = st.sidebar.number_input("Page", min_value=1, value=1, key="notes-page")
    offset = (page - 1) * limit

    summary = get_notes(limit=limit, offset=offset)
    notes: list[dict[str, Any]] = summary["notes"]
    total = summary["total"]

    tag_options = sorted(
        {
            tag_value
            for note in notes
            for tag in note.get("tags", [])
            if isinstance(tag, dict)
            for tag_value in [
                tag.get("TAG") or tag.get("tag"),
            ]
            if isinstance(tag_value, str) and tag_value.strip()
        }
    )
    tag_filter = st.sidebar.multiselect(
        "Filter by tag",
        options=tag_options,
        placeholder="Select tags",
        key="notes-tag-filter",
    )
    notes = _filter_notes_by_tags(notes, tag_filter)

    tag_batch_limit = st.sidebar.slider(
        "Tagging batch size",
        min_value=1,
        max_value=100,
        value=25,
        key="tagging-batch-size",
    )
    if st.sidebar.button("Tag untagged notes", use_container_width=True):
        with st.spinner("Classifying tags for untagged notes..."):
            response = run_tagging_job(tag_batch_limit)
        if response and isinstance(response, dict):
            tagged = response.get("tagged", [])
            skipped = response.get("skipped", [])
            st.sidebar.success(
                f"Tagged {len(tagged)} notes." if tagged else "No new tags applied."
            )
            if skipped:
                st.sidebar.warning(
                    f"Skipped {len(skipped)} notes: " + ", ".join(skipped)
                )
        else:
            st.sidebar.error("Unable to run tagging. Check logs for details.")

    if not notes:
        st.info("No notes found. Try creating one from the **Create** tab.")
        return

    sorted_notes = _sort_notes_by_created(notes)
    grouped_notes = _group_notes_by_context(sorted_notes)

    for index, (context_label, context_notes) in enumerate(grouped_notes):
        st.markdown(f"### {context_label}")
        for note in context_notes:
            _render_note_panel(note)
        if index < len(grouped_notes) - 1:
            st.divider()

    st.caption(f"Showing {len(notes)} of {total} notes.")


def _context_color(label: str) -> str:
    mapping = {
        "work": "#0070c9",
        "personal": "#9b59b6",
        "leisure": "#2ecc71",
        "mixed": "#e67e22",
        "unknown": "#7f8c8d",
        "pending": "#bdc3c7",
    }
    key = _normalize_context_key(label)
    return mapping.get(key, "#34495e")


def _escape_graphviz_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\"", r"\"")
        .replace("\n", "\\n")
        .strip()
    )


def _collect_shared_tags(notes: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tags_map: dict[str, list[dict[str, Any]]] = {}
    for note in notes:
        tags = note.get("tags") or []
        for tag in tags:
            tag_value = tag.get("TAG") or tag.get("tag")
            if isinstance(tag_value, str) and tag_value.strip():
                normalized = tag_value.strip().lower()
                tags_map.setdefault(normalized, []).append(note)
    return {tag: items for tag, items in tags_map.items() if len(items) >= 2}


def _prepare_graph_nodes(
    notes: Iterable[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    node_labels: dict[str, str] = {}
    node_colors: dict[str, str] = {}
    for note in notes:
        note_id = str(note.get("note_id") or "")
        if not note_id:
            continue
        title = note.get("title") or "Untitled"
        context_label = note.get("context_label") or note.get("sentiment_label") or "pending"
        confidence = note.get("context_confidence")
        if confidence is None:
            confidence = note.get("sentiment_score")
        context_summary = _format_context_summary(context_label, confidence)
        label_text = f"{title[:40]}{'…' if len(title) > 40 else ''}\\n{context_summary}"
        node_labels[note_id] = _escape_graphviz_text(label_text)
        node_colors[note_id] = _context_color(context_label)
    return node_labels, node_colors


def _generate_edges(shared_tags: dict[str, list[dict[str, Any]]]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for tag_notes in shared_tags.values():
        for i in range(len(tag_notes)):
            for j in range(i + 1, len(tag_notes)):
                first = str(tag_notes[i].get("note_id") or "")
                second = str(tag_notes[j].get("note_id") or "")
                if first and second and first != second:
                    edge = (first, second) if first < second else (second, first)
                    edges.add(edge)
    return edges


def _build_relationship_graph(
    notes: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    shared_tags = _collect_shared_tags(notes)
    if not shared_tags:
        return None, []

    node_labels, node_colors = _prepare_graph_nodes(notes)
    edges = _generate_edges(shared_tags)

    if not edges:
        return None, []

    dot_lines: list[str] = [
        "digraph NotesGraph {",
        "  graph [splines=true overlap=false fontname=\"Helvetica\"];",
        "  node [shape=box style=filled fontname=\"Helvetica\" fontsize=12];",
    ]

    for node_id, label in node_labels.items():
        color = node_colors.get(node_id, "#34495e")
        dot_lines.append(f'  "{node_id}" [label="{label}" fillcolor="{color}" fontcolor="#ffffff"];')

    for source, target in edges:
        dot_lines.append(f'  "{source}" -> "{target}" [dir=both color="#95a5a6"];')

    dot_lines.append("}")

    tag_summary = [
        {"Tag": tag, "Notes": len(tag_notes)}
        for tag, tag_notes in sorted(shared_tags.items(), key=lambda item: len(item[1]), reverse=True)
    ]
    return "\n".join(dot_lines), tag_summary

def render_relations() -> None:
    st.subheader("Note Relationship Graph")
    max_notes = st.slider("Notes to consider", min_value=20, max_value=200, value=100, step=10)
    summary = get_notes(limit=max_notes, offset=0)
    notes: list[dict[str, Any]] = summary["notes"]

    if not notes:
        st.info("No notes available to visualize yet.")
        return

    st.caption(
        "Nodes represent notes coloured by context. Edges connect notes that share at least one tag."
    )

    dot_graph, tag_summary = _build_relationship_graph(notes)
    if not dot_graph:
        st.warning("Not enough shared tags between notes to build a relationship graph.")
        return

    st.graphviz_chart(dot_graph, use_container_width=True)

    if tag_summary:
        st.markdown("#### Shared Tags")
        st.dataframe(tag_summary, use_container_width=True, hide_index=True)


def render_search() -> None:
    st.subheader("Semantic Search")
    query = st.session_state.get("search_query", "")
    if not query:
        st.info("Use the sidebar search box to explore your memories.")
        return

    st.caption(f"Results for **{query}**")
    k = st.slider(
        "Results",
        min_value=3,
        max_value=25,
        value=st.session_state.get("search_results_limit", 8),
        key="search_results_limit",
    )

    with st.spinner("Searching..."):
        results = search_notes(query=query, limit=k)
    hits = results.get("results", [])
    if not hits:
        st.warning("No matches found. Try refining your query.")
        return

    for hit in hits:
        distance_value = float(hit.get("distance", 0.0) or 0.0)
        with st.container():
            badge_html = _render_similarity_badge(distance_value)
            st.markdown(
                f"**Note:** {hit.get('note_id', 'unknown')} &nbsp; · &nbsp; "
                f"**Chunk:** {hit.get('chunk_index', 0)} &nbsp; · &nbsp; "
                f"{badge_html}",
                unsafe_allow_html=True,
            )
            st.write(hit.get("text", ""))
            st.divider()
    st.session_state.setdefault("recent_searches", []).insert(
        0,
        {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "hits": len(hits),
        },
    )


def _get_stat_value(stats: dict[str, Any], key: str, default: Any = 0) -> Any:
    if key in stats:
        return stats.get(key, default)
    return stats.get(key.upper(), default)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_context_label(value: Any) -> str:
    if value is None:
        return "Pending"
    label = str(value).strip()
    if not label:
        return "Pending"
    return label.replace("_", " ").title()


def _normalize_context_key(value: Any) -> str:
    if value is None:
        return "pending"
    label = str(value).strip().lower()
    if not label:
        return "pending"
    return label


def _format_context_summary(label: Any, confidence: Any) -> str:
    label_text = _format_context_label(label)
    confidence_value = _coerce_float(confidence)
    if confidence_value is not None:
        return f"{label_text} ({confidence_value:.2f})"
    return label_text


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _sort_notes_by_created(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        notes,
        key=lambda note: _parse_datetime(note.get("created_at")) or datetime.min,
        reverse=True,
    )


def _group_notes_by_context(
    notes: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for note in notes:
        key = _normalize_context_key(
            note.get("context_label") or note.get("sentiment_label")
        )
        grouped.setdefault(key, []).append(note)

    context_order = [
        "work",
        "personal",
        "leisure",
        "mixed",
        "unknown",
        "pending",
    ]
    seen = set(context_order)
    for key in grouped:
        if key not in seen:
            context_order.append(key)
            seen.add(key)

    ordered: list[tuple[str, list[dict[str, Any]]]] = []
    for key in context_order:
        bucket = grouped.get(key, [])
        if bucket:
            ordered.append((_format_context_label(key), bucket))
    return ordered


def _render_context_metrics(stats: dict[str, Any]) -> None:
    total_notes = int(_get_stat_value(stats, "total_notes", 0) or 0)
    work_notes = int(_get_stat_value(stats, "work_notes", 0) or 0)
    personal_notes = int(_get_stat_value(stats, "personal_notes", 0) or 0)
    leisure_notes = int(_get_stat_value(stats, "leisure_notes", 0) or 0)
    mixed_notes = int(_get_stat_value(stats, "mixed_notes", 0) or 0)
    unknown_notes = int(_get_stat_value(stats, "unknown_notes", 0) or 0)
    pending_notes = int(_get_stat_value(stats, "pending_notes", 0) or 0)
    average_confidence = _get_stat_value(stats, "average_context_confidence", None)
    last_analyzed_at_raw = _get_stat_value(stats, "last_analyzed_at", None)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analyzed Notes", f"{total_notes:,}")
    col2.metric("Work", f"{work_notes:,}")
    col3.metric("Personal", f"{personal_notes:,}")
    col4.metric("Leisure", f"{leisure_notes:,}")

    extra_col1, extra_col2, extra_col3 = st.columns(3)
    extra_col1.metric("Mixed", f"{mixed_notes:,}")
    extra_col2.metric("Unknown", f"{unknown_notes:,}")
    extra_col3.metric(
        "Pending Review",
        f"{pending_notes:,}",
        help="Notes still awaiting context classification.",
    )

    if average_confidence is not None:
        st.metric(
            "Average Confidence",
            f"{float(average_confidence):.3f}",
            help="Mean classification confidence across analyzed notes.",
        )
    if last_analyzed_at_raw:
        st.caption(
            f"Last context refresh: {format_timestamp(str(last_analyzed_at_raw))}"
        )


def _render_context_job(job: dict[str, Any], job_id: str) -> bool:
    status_value = job.get("status", "unknown")
    st.write(f"**Job ID:** `{job_id}`  \n**Status:** {status_value.title()}")
    summary = job.get("summary") or {}
    if summary:
        top_cols = st.columns(3)
        top_cols[0].metric("Processed", summary.get("processed", 0))
        top_cols[1].metric("Skipped", summary.get("skipped", 0))
        top_cols[2].metric("Failed", summary.get("failed", 0))

        context_cols = st.columns(4)
        context_cols[0].metric("Work", summary.get("work_notes", 0))
        context_cols[1].metric("Personal", summary.get("personal_notes", 0))
        context_cols[2].metric("Leisure", summary.get("leisure_notes", 0))
        context_cols[3].metric("Mixed", summary.get("mixed_notes", 0))
        st.metric(
            "Unknown",
            summary.get("unknown_notes", 0),
            help="Notes that could not be clearly categorized.",
        )
        avg_conf = summary.get("average_context_confidence")
        if avg_conf is not None:
            st.metric(
                "Average Confidence",
                f"{float(avg_conf):.3f}",
                help="Mean confidence for this job run.",
            )
    started_at = format_timestamp(job.get("started_at"))
    completed_at = format_timestamp(job.get("completed_at"))
    st.caption(f"Started: {started_at} · Completed: {completed_at}")

    logs = job.get("logs", [])
    if logs:
        st.markdown("#### Job Log")
        for entry in logs:
            status_text = entry.get("status", "").title()
            note_id = entry.get("note_id", "unknown")
            label = entry.get("label")
            score = entry.get("score")
            message = entry.get("message")
            details: list[str] = []
            if label:
                details.append(f"Context: {label}")
            if score is not None:
                details.append(f"Confidence: {score:.3f}")
            if message:
                details.append(message)
            detail_text = " · ".join(details)
            st.write(f"- `{note_id}` — {status_text}" + (f" ({detail_text})" if detail_text else ""))
    else:
        st.info("No log entries yet for this job.")

    if status_value == "completed":
        st.success("Context classification job completed.")
        return True
    if status_value == "failed":
        st.error(f"Context classification job failed: {job.get('error', 'Unknown error')}")
        return True
    return False


def _render_context_history(jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        st.info("No context classification jobs recorded yet.")
        return

    history = []
    for job in jobs[:20]:
        summary = job.get("summary") or {}
        history.append(
            {
                "Job ID": job.get("job_id"),
                "Status": job.get("status", "").title(),
                "Processed": summary.get("processed"),
                "Skipped": summary.get("skipped"),
                "Failed": summary.get("failed"),
                "Work": summary.get("work_notes"),
                "Personal": summary.get("personal_notes"),
                "Leisure": summary.get("leisure_notes"),
                "Mixed": summary.get("mixed_notes"),
                "Unknown": summary.get("unknown_notes"),
                "Avg Confidence": summary.get("average_context_confidence"),
                "Started": format_timestamp(job.get("started_at")),
                "Completed": format_timestamp(job.get("completed_at")),
            }
        )
    st.dataframe(history, use_container_width=True)


def render_context() -> None:
    st.subheader("Context Analytics")
    stats = get_sentiment_stats()
    _render_context_metrics(stats)

    st.markdown("### Run Context Classification")
    default_limit = (
        st.session_state.get("context_limit")
        or st.session_state.get("sentiment_limit")
        or 100
    )
    limit = st.slider(
        "Notes per batch",
        min_value=10,
        max_value=500,
        value=default_limit,
        step=10,
        key="context-limit-slider",
    )
    st.session_state["context_limit"] = limit
    st.session_state.pop("sentiment_limit", None)

    if st.button("Run context analysis now", type="primary"):
        response = run_sentiment_job(limit)
        if response and "job_id" in response:
            job_id = response["job_id"]
            st.session_state["context_active_job"] = job_id
            st.success(f"Context classification job `{job_id}` started.")
        else:
            st.error("Unable to start the context classification job. Please try again.")

    active_job_id = st.session_state.get("context_active_job") or st.session_state.get(
        "sentiment_active_job"
    )
    if active_job_id:
        st.markdown("### Active Job")
        job = get_sentiment_job(active_job_id)
        if job:
            finished = _render_context_job(job, active_job_id)
            if finished:
                st.cache_data.clear()
                st.session_state.pop("context_active_job", None)
                st.session_state.pop("sentiment_active_job", None)
        else:
            st.warning("Unable to retrieve job status. Try refreshing.")
    st.markdown("### Recent Context Jobs")
    _render_context_history(list_sentiment_jobs())


def render_create_note() -> None:
    LOGGER.info("render_create_note invoked")
    st.subheader("Capture a New Memory")
    st.session_state.setdefault("create_title", "")
    st.session_state.setdefault("create_user_id", "user-1")
    st.session_state.setdefault("create_source", "streamlit")
    st.session_state.setdefault("create_content", "")
    st.session_state.setdefault("create_tags_raw", "")
    if st.session_state.pop("create_reset_form", False):
        st.session_state["create_title"] = ""
        st.session_state["create_user_id"] = "user-1"
        st.session_state["create_source"] = "streamlit"
        st.session_state["create_content"] = ""
        st.session_state["create_tags_raw"] = ""

    title = st.text_input(
        "Title",
        placeholder="Weekly planning recap",
        key="create_title",
    )
    user_id = st.text_input("User ID", key="create_user_id")
    source = st.text_input("Source", key="create_source")
    content = st.text_area(
        "Content",
        height=200,
        key="create_content",
        placeholder="Write the details you want Atsiitsʼiin to remember...",
    )
    tags_raw = st.text_input(
        "Tags (comma or newline separated)",
        key="create_tags_raw",
        placeholder="project-phoenix, urgent, quarterly-planning",
        help="Optional. Provide one or more tags to pin to this note. "
        "We will keep your tags even when suggestions refresh.",
    )
    user_tags_preview = _parse_user_tag_input(tags_raw)
    if user_tags_preview:
        preview_payload = [
            {
                "tag": tag["tag"],
                "display_name": tag["display_name"],
                "source": "user",
            }
            for tag in user_tags_preview
        ]
        st.markdown(
            _render_tag_pills(preview_payload),
            unsafe_allow_html=True,
        )
    submit_clicked = st.button(
        "Save to Snowflake", type="primary", key="create_submit"
    )
    LOGGER.info(
        "Create button render clicked=%s title_len=%d content_len=%d",
        submit_clicked,
        len(title or ""),
        len(content or ""),
    )

    if submit_clicked:
        LOGGER.info(
            "Create button clicked title=%r user_id=%r source=%r content_length=%d",
            title,
            user_id,
            source,
            len(content or ""),
        )
        if not title or not title.strip() or not content or not content.strip():
            st.error("Title and content are required.")
            LOGGER.warning("Create note aborted: missing title or content")
        else:
            with st.spinner("Saving note..."):
                note_id = create_note(
                    title.strip(),
                    content.strip(),
                    user_id.strip(),
                    source.strip(),
                    tags=user_tags_preview,
                )
            LOGGER.info("Create note result note_id=%s title=%r", note_id, title)
            if note_id:
                st.success(f"Saved! Note ID: `{note_id}`")
                st.session_state.setdefault("recent_activity", []).insert(
                    0,
                    {
                        "type": "note_created",
                        "title": title,
                        "note_id": note_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
                st.session_state["create_reset_form"] = True
                st.rerun()


def render_activity() -> None:
    st.subheader("Recent Activity")
    recent_activity = st.session_state.get("recent_activity", [])
    recent_searches = st.session_state.get("recent_searches", [])

    if not recent_activity and not recent_searches:
        st.info("Interact with the app to see activity here.")
        return

    if recent_activity:
        st.markdown("### Captures & Actions")
        for item in recent_activity[:10]:
            st.write(
                f"✅ **Created note** `{item['note_id']}` — {item['title']} "
                f"({format_timestamp(item['timestamp'])})"
            )
    if recent_searches:
        st.markdown("### Searches")
        for entry in recent_searches[:10]:
            st.write(
                f"🔎 **{entry['query']}** — {entry['hits']} results "
                f"({format_timestamp(entry['timestamp'])})"
            )


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --background-color: #f4f6f8;
                --card-color: #ffffff;
                --text-color: #1c1c1e;
                --text-color-muted: #6e6e73;
                --accent-color: #0070c9;
                --accent-color-light: #1e90ff;
            }

            .stApp {
                background-color: var(--background-color);
                color: var(--text-color);
            }

            .stButton>button {
                border-radius: 999px;
                padding: 0.5rem 1.5rem;
                border: none;
                background: linear-gradient(135deg, var(--accent-color), var(--accent-color-light));
                color: white;
                font-weight: 600;
                box-shadow: 0 10px 20px rgba(0, 112, 201, 0.2);
            }

            .stTextInput>div>div>input,
            .stTextArea>div>textarea {
                border-radius: 18px;
                border: 1px solid rgba(0,0,0,0.1);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 4rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_navigation(nav_options: list[str]) -> tuple[str, bool, str]:
    with st.sidebar:
        if (
            "sidebar_search_input" not in st.session_state
            and st.session_state.get("search_query")
        ):
            st.session_state["sidebar_search_input"] = st.session_state["search_query"]

        search_value = st.text_input(
            label="Search memories",
            placeholder="Search your second brain...",
            key="sidebar_search_input",
        )
        normalized_query = search_value.strip()
        if normalized_query:
            if st.button("Clear search", key="sidebar-clear-search", use_container_width=True):
                st.session_state["sidebar_search_input"] = ""
                st.session_state.pop("search_query", None)
                normalized_query = ""
        else:
            normalized_query = ""

        st.header("Navigation")
        current_radio = st.session_state.get("nav_radio_value", nav_options[0])
        if current_radio == "Sentiment":
            current_radio = "Context"
            st.session_state["nav_radio_value"] = current_radio
        if current_radio not in nav_options:
            current_radio = nav_options[0]
            st.session_state["nav_radio_value"] = current_radio
        selected = st.radio(
            label="Select view",
            options=nav_options,
            index=nav_options.index(current_radio),
            label_visibility="collapsed",
        )
        create_clicked = st.button("➕ Create Note", use_container_width=True)
        st.markdown("---")
        st.markdown(
            f"""
            **API Endpoint**
            `{API_BASE_URL}`
            """
        )

    return selected, create_clicked, normalized_query


def _render_similarity_badge(distance: float) -> str:
    similarity = max(0.0, min(1.0, 1.0 - distance))
    pct = similarity * 100
    if pct >= 85:
        bg_color = "#2ecc71"
        text_color = "#ffffff"
    elif pct >= 65:
        bg_color = "#f1c40f"
        text_color = "#1c1c1e"
    elif pct >= 45:
        bg_color = "#e67e22"
        text_color = "#ffffff"
    else:
        bg_color = "#e74c3c"
        text_color = "#ffffff"

    return (
        "<span style=\"display:inline-flex;align-items:center;gap:0.35rem;\">"
        f"<span style=\"display:inline-block;padding:0.15rem 0.6rem;"
        f"border-radius:999px;background:{bg_color};color:{text_color};"
        f"font-size:0.75rem;font-weight:600;letter-spacing:0.01em;\">"
        f"{pct:.0f}% match</span>"
        f"<span style=\"font-size:0.75rem;color:var(--text-color-muted);\">"
        f"(distance {distance:.3f})</span>"
        "</span>"
    )


def resolve_navigation() -> str:
    if "nav_view" not in st.session_state:
        st.session_state["nav_view"] = "Dashboard"
    if "nav_radio_value" not in st.session_state:
        st.session_state["nav_radio_value"] = "Dashboard"

    if (
        st.session_state.get("nav_view") == "Search"
        and not st.session_state.get("search_query")
    ):
        st.session_state["nav_view"] = st.session_state.get("nav_radio_value", "Dashboard")

    nav_options = ["Dashboard", "Notes", "Context", "Relations", "Activity"]
    selected, create_clicked, normalized_query = _render_sidebar_navigation(nav_options)

    previous_view = st.session_state.get("nav_view", "Dashboard")
    if previous_view == "Sentiment":
        previous_view = "Context"
        st.session_state["nav_view"] = "Context"
    search_active = bool(normalized_query)
    if search_active:
        st.session_state["search_query"] = normalized_query
        st.session_state["nav_view"] = "Search"
    else:
        st.session_state.pop("search_query", None)

    if create_clicked:
        st.session_state["nav_view"] = "Create"
    else:
        st.session_state["nav_radio_value"] = selected
        if search_active:
            st.session_state["nav_view"] = "Search"
        elif previous_view != "Create":
            st.session_state["nav_view"] = selected

    return st.session_state["nav_view"]


def main() -> None:
    st.set_page_config(
        page_title="Atsiitsʼiin — Second Brain",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_global_styles()

    st.title("🧠 Atsiitsʼiin")
    st.caption("Your agentic Snowflake-backed second brain.")

    view = resolve_navigation()

    if view == "Dashboard":
        render_dashboard()
    elif view == "Notes":
        render_notes_list()
    elif view == "Search":
        render_search()
    elif view == "Create":
        render_create_note()
    elif view == "Context":
        render_context()
    elif view == "Relations":
        render_relations()
    elif view == "Activity":
        render_activity()


if __name__ == "__main__":
    main()
