from typing import Any

from ..config import AtsiiitsiinConfig
from ..memory import ingest_note, search_memory
from ..pipelines.tagging import TaggingError, classify_note_tags


def tool_add_note(
    args: dict[str, Any], cfg: AtsiiitsiinConfig
) -> dict[str, Any]:
    user_id = args.get("user_id", "user-1")
    source = args.get("source", "manual")
    title = args["title"]
    content = args["content"]
    note_id = ingest_note(user_id, source, title, content, cfg)
    response: dict[str, Any] = {"note_id": note_id}

    try:
        tagging_result = classify_note_tags(note_id, cfg, source="llm_suggestion")
        response["tags"] = [tag.to_dict() for tag in tagging_result.tags]
        response["audit_id"] = tagging_result.audit_id
    except TaggingError as exc:
        response["tagging_error"] = str(exc)

    return response


def tool_search(
    args: dict[str, Any], cfg: AtsiiitsiinConfig
) -> list[dict[str, Any]]:
    query = args["query"]
    k = int(args.get("k", 8))
    return search_memory(query, cfg, k)


def tool_add_tags(
    args: dict[str, Any], cfg: AtsiiitsiinConfig
) -> dict[str, Any]:
    note_id = args["note_id"]
    max_tags = int(args.get("max_tags", 5))
    source = args.get("source", "llm_suggestion")

    tagging_result = classify_note_tags(
        note_id,
        cfg,
        max_tags=max_tags,
        source=source,
    )
    return {
        "note_id": note_id,
        "tags": [tag.to_dict() for tag in tagging_result.tags],
        "audit_id": tagging_result.audit_id,
    }
