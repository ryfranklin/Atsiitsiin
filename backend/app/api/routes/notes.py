from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from atsiitsiin.config import AtsiiitsiinConfig

from ...core.config import get_memory_config
from ...schemas import notes as notes_schema
from ...services import memory as memory_service
from ...services import sentiment as sentiment_service

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post(
    "",
    response_model=notes_schema.NoteCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new note",
)
async def create_note_endpoint(
    payload: notes_schema.NoteCreateRequest,
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.NoteCreateResponse:
    note_id = await memory_service.create_note(payload, config)
    return notes_schema.NoteCreateResponse(note_id=note_id)


@router.get(
    "",
    response_model=notes_schema.NoteListResponse,
    summary="List recent notes",
)
async def list_notes_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.NoteListResponse:
    notes, total = await memory_service.fetch_notes(config, limit=limit, offset=offset)
    summaries: list[notes_schema.NoteSummary] = [
        notes_schema.NoteSummary.model_validate(note) for note in notes
    ]
    return notes_schema.NoteListResponse(notes=summaries, total=total)


@router.get(
    "/search",
    response_model=notes_schema.NoteSearchResponse,
    summary="Search notes using semantic similarity",
)
async def search_notes_endpoint(
    q: str = Query(..., min_length=1, max_length=512),
    k: int = Query(8, ge=1, le=25),
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.NoteSearchResponse:
    hits = await memory_service.search_notes(query=q, config=config, limit=k)
    results = [notes_schema.NoteSearchHit.model_validate(hit) for hit in hits]
    return notes_schema.NoteSearchResponse(query=q, results=results)


@router.get(
    "/{note_id}",
    response_model=notes_schema.NoteDetail,
    summary="Retrieve a note and optional chunks",
)
async def get_note_endpoint(
    note_id: str,
    include_chunks: bool = Query(False),
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.NoteDetail:
    note = await memory_service.fetch_note(
        note_id=note_id, config=config, include_chunks=include_chunks
    )
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found",
        )
    return notes_schema.NoteDetail.model_validate(note)


@router.get(
    "/analytics/sentiment",
    summary="Get sentiment analysis summary",
)
async def sentiment_analytics_endpoint(
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> dict[str, Any]:
    summary = await memory_service.get_sentiment_stats(config)
    return summary


@router.post(
    "/analytics/sentiment/run",
    response_model=notes_schema.SentimentJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger asynchronous sentiment analysis refresh",
)
async def sentiment_run_endpoint(
    payload: Annotated[notes_schema.SentimentRunRequest | None, Body()] = None,
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.SentimentJobStatus:
    limit = payload.limit if payload else 100
    state = sentiment_service.start_job(config, limit=limit)
    return _state_to_status(state)


@router.post(
    "/analytics/tagging/run",
    response_model=notes_schema.TaggingRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Classify tags for untagged notes",
)
async def tagging_run_endpoint(
    payload: Annotated[notes_schema.TaggingRunRequest | None, Body()] = None,
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.TaggingRunResponse:
    limit = payload.limit if payload else 25
    note_ids = payload.note_ids if payload else None
    tagged, skipped = await memory_service.run_tagging(
    config, limit=limit, note_ids=note_ids
    )
    results: list[notes_schema.TaggingResult] = []
    for item in tagged:
        results.append(
            notes_schema.TaggingResult(
                note_id=item["note_id"],
                audit_id=item.get("audit_id"),
                tags=[
                    notes_schema.TagSuggestion.model_validate(tag)
                    for tag in item["tags"]
                ],
            )
        )
    return notes_schema.TaggingRunResponse(tagged=results, skipped=skipped)


@router.get(
    "/analytics/sentiment/jobs",
    response_model=notes_schema.SentimentJobListResponse,
    summary="List sentiment analysis jobs",
)
async def sentiment_jobs_list_endpoint() -> notes_schema.SentimentJobListResponse:
    states = sentiment_service.list_jobs()
    return notes_schema.SentimentJobListResponse(
        jobs=[_state_to_status(state) for state in states]
    )


@router.get(
    "/analytics/sentiment/jobs/{job_id}",
    response_model=notes_schema.SentimentJobStatus,
    summary="Get sentiment analysis job detail",
)
async def sentiment_job_detail_endpoint(job_id: str) -> notes_schema.SentimentJobStatus:
    state = sentiment_service.get_job(job_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sentiment job not found",
        )
    return _state_to_status(state)


@router.post(
    "/{note_id}/tags",
    response_model=notes_schema.NoteTagUpdateResponse,
    summary="Add user tags to a note",
)
async def add_note_tags_endpoint(
    note_id: str,
    payload: notes_schema.NoteTagUpdateRequest,
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.NoteTagUpdateResponse:
    if not payload.tags:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one tag is required",
        )
    try:
        updated = await memory_service.apply_user_tags(note_id, payload.tags, config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    tags = [
        notes_schema.NoteTag.model_validate(tag) for tag in updated
    ]
    return notes_schema.NoteTagUpdateResponse(note_id=note_id, tags=tags)


@router.delete(
    "/{note_id}/tags",
    response_model=notes_schema.NoteTagRemoveResponse,
    summary="Remove tags from a note",
)
async def remove_note_tags_endpoint(
    note_id: str,
    payload: Annotated[notes_schema.NoteTagRemoveRequest, Body()],
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.NoteTagRemoveResponse:
    if not payload.tags:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one tag is required to remove",
        )
    try:
        remaining = await memory_service.remove_note_tags(
            note_id, payload.tags, config
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    tags = [
        notes_schema.NoteTag.model_validate(tag) for tag in remaining
    ]
    return notes_schema.NoteTagRemoveResponse(note_id=note_id, tags=tags)


@router.delete(
    "/{note_id}",
    response_model=notes_schema.NoteDeleteResponse,
    summary="Delete a note",
)
async def delete_note_endpoint(
    note_id: str,
    config: AtsiiitsiinConfig = Depends(get_memory_config),  # noqa: B008
) -> notes_schema.NoteDeleteResponse:
    try:
        await memory_service.delete_note(note_id, config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return notes_schema.NoteDeleteResponse(note_id=note_id, deleted=True)


def _state_to_status(
    state: sentiment_service.SentimentJobState,
) -> notes_schema.SentimentJobStatus:
    summary = None
    if state.result is not None:
        summary = notes_schema.SentimentJobSummary(
            processed=state.result.processed,
            skipped=state.result.skipped,
            failed=state.result.failed,
            work_notes=state.result.work_notes,
            personal_notes=state.result.personal_notes,
            leisure_notes=state.result.leisure_notes,
            mixed_notes=state.result.mixed_notes,
            unknown_notes=state.result.unknown_notes,
            average_context_confidence=state.result.average_context_confidence,
        )
    logs = [notes_schema.SentimentJobLog.model_validate(entry) for entry in state.logs]
    return notes_schema.SentimentJobStatus(
        job_id=state.job_id,
        status=state.status,
        started_at=state.started_at,
        completed_at=state.completed_at,
        summary=summary,
        logs=logs,
        error=state.error,
    )

