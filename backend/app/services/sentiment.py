# ruff: noqa: BLE001
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from atsiitsiin.config import AtsiiitsiinConfig
from atsiitsiin.pipelines.sentiment import (
    SentimentJobLogEntry,
    SentimentJobResult,
    run_sentiment_job,
)

LOGGER = logging.getLogger("atsiitsiin.backend.sentiment")
# ruff: noqa: BLE001


@dataclass
class SentimentJobState:
    job_id: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    completed_at: datetime | None = None
    result: SentimentJobResult | None = None
    logs: list[SentimentJobLogEntry] = field(default_factory=list)
    error: str | None = None


_jobs: dict[str, SentimentJobState] = {}
_lock = threading.Lock()


class SentimentJobExecutionError(RuntimeError):
    """Raised when the sentiment job run fails unexpectedly."""


def start_job(config: AtsiiitsiinConfig, limit: int = 100) -> SentimentJobState:
    job_id = str(uuid4())
    state = SentimentJobState(
        job_id=job_id,
        status="running",
        started_at=datetime.now(UTC),
    )

    with _lock:
        _jobs[job_id] = state

    def _on_entry(entry: SentimentJobLogEntry) -> None:
        with _lock:
            state.logs.append(entry)

    def _runner() -> None:
        try:
            result = run_sentiment_job(
                limit=limit,
                cfg=config,
                logger=LOGGER,
                on_entry=_on_entry,
            )
            with _lock:
                state.result = result
                state.status = "completed"
                state.completed_at = datetime.now(UTC)
                if not state.logs:
                    state.logs.extend(result.logs)
        except Exception as exc:
            LOGGER.exception("Sentiment job %s failed", job_id)
            with _lock:
                state.status = "failed"
                state.error = str(exc)
                state.completed_at = datetime.now(UTC)
            raise SentimentJobExecutionError(str(exc)) from exc

    thread = threading.Thread(target=_runner, name=f"sentiment-job-{job_id}", daemon=True)
    thread.start()
    return snapshot_state(job_id)


def get_job(job_id: str) -> SentimentJobState | None:
    with _lock:
        if job_id not in _jobs:
            return None
    return snapshot_state(job_id)


def list_jobs() -> list[SentimentJobState]:
    with _lock:
        job_ids = list(_jobs.keys())
    states = [snapshot_state(job_id) for job_id in job_ids]
    states.sort(key=lambda s: s.started_at, reverse=True)
    return states


def snapshot_state(job_id: str) -> SentimentJobState:
    with _lock:
        state = _jobs[job_id]
        copied_logs = list(state.logs)
        return SentimentJobState(
            job_id=state.job_id,
            status=state.status,
            started_at=state.started_at,
            completed_at=state.completed_at,
            result=state.result,
            logs=copied_logs,
            error=state.error,
        )
