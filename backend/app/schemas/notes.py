from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NoteChunk(BaseModel):
    chunk_id: str = Field(alias="ID")
    note_id: str = Field(alias="NOTE_ID")
    chunk_index: int = Field(alias="CHUNK_INDEX")
    text: str = Field(alias="TEXT")
    embedding: list[float] | None = Field(
        default=None, alias="EMBEDDING", description="Embedding vector when available."
    )

    class Config:
        populate_by_name = True


class NoteTag(BaseModel):
    tag: str = Field(alias="TAG")
    display_name: str | None = Field(alias="DISPLAY_NAME", default=None)
    description: str | None = Field(alias="DESCRIPTION", default=None)
    color_hex: str | None = Field(alias="COLOR_HEX", default=None)
    source: str | None = Field(alias="SOURCE", default=None)
    confidence: float | None = Field(alias="CONFIDENCE", default=None)

    class Config:
        populate_by_name = True


class NoteSummary(BaseModel):
    note_id: str = Field(alias="ID")
    user_id: str = Field(alias="USER_ID")
    source: str = Field(alias="SOURCE")
    title: str = Field(alias="TITLE")
    created_at: datetime | None = Field(alias="CREATED_AT", default=None)
    context_label: str | None = Field(alias="CONTEXT_LABEL", default=None)
    context_confidence: float | None = Field(alias="CONTEXT_CONFIDENCE", default=None)
    context_analyzed_at: datetime | None = Field(
        alias="CONTEXT_ANALYZED_AT", default=None
    )
    sentiment_label: str | None = Field(alias="SENTIMENT_LABEL", default=None)
    sentiment_score: float | None = Field(alias="SENTIMENT_SCORE", default=None)
    sentiment_analyzed_at: datetime | None = Field(
        alias="SENTIMENT_ANALYZED_AT", default=None
    )
    tags: list[NoteTag] = Field(default_factory=list, alias="tags")

    class Config:
        populate_by_name = True


class NoteDetail(NoteSummary):
    description: str | None = Field(alias="DESCRIPTION", default=None)
    chunks: list[NoteChunk] = Field(default_factory=list)


class NoteTagInput(BaseModel):
    tag: str = Field(..., min_length=1)
    display_name: str | None = None
    description: str | None = None
    color_hex: str | None = None
    confidence: float | None = None


class NoteCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=1)
    user_id: str = Field(default="user-1", min_length=1)
    source: str = Field(default="manual", min_length=1)
    tags: list[NoteTagInput] | None = None


class NoteCreateResponse(BaseModel):
    note_id: str


class NoteTagUpdateRequest(BaseModel):
    tags: list[NoteTagInput] = Field(default_factory=list)


class NoteTagUpdateResponse(BaseModel):
    note_id: str
    tags: list[NoteTag]


class NoteTagRemoveRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)


class NoteTagRemoveResponse(BaseModel):
    note_id: str
    tags: list[NoteTag]


class NoteDeleteResponse(BaseModel):
    note_id: str
    deleted: bool


class NoteListResponse(BaseModel):
    notes: list[NoteSummary]
    total: int


class NoteSearchHit(BaseModel):
    note_id: str = Field(alias="NOTE_ID")
    chunk_id: str = Field(alias="ID")
    chunk_index: int = Field(alias="CHUNK_INDEX")
    text: str = Field(alias="TEXT")
    distance: float

    class Config:
        populate_by_name = True


class NoteSearchResponse(BaseModel):
    query: str
    results: list[NoteSearchHit]


class SentimentRunRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=500)


class SentimentJobLog(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    note_id: str
    status: str
    message: str | None = None
    label: str | None = None
    score: float | None = None


class SentimentJobSummary(BaseModel):
    processed: int
    skipped: int
    failed: int
    work_notes: int | None = None
    personal_notes: int | None = None
    leisure_notes: int | None = None
    mixed_notes: int | None = None
    unknown_notes: int | None = None
    average_context_confidence: float | None = None


class SentimentJobStatus(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    completed_at: datetime | None = None
    summary: SentimentJobSummary | None = None
    logs: list[SentimentJobLog] = Field(default_factory=list)
    error: str | None = None


class SentimentJobListResponse(BaseModel):
    jobs: list[SentimentJobStatus]


class TagSuggestion(BaseModel):
    tag: str
    display_name: str | None = None
    description: str | None = None
    color_hex: str | None = None
    confidence: float | None = None
    source: str | None = None


class TaggingResult(BaseModel):
    note_id: str
    tags: list[TagSuggestion]
    audit_id: str | None = None


class TaggingRunRequest(BaseModel):
    limit: int = Field(default=25, ge=1, le=200)
    note_ids: list[str] | None = None


class TaggingRunResponse(BaseModel):
    tagged: list[TaggingResult]
    skipped: list[str] = Field(default_factory=list)

