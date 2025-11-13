"""Snowpark sentiment analysis pipeline for Atsiitsʼiin."""

from __future__ import annotations

import logging

from atsiitsiin.pipelines.sentiment import (
    SentimentJobLogEntry,
    run_sentiment_job,
)

LOGGER = logging.getLogger("atsiitsiin.pipeline.sentiment")


def _log_entry(logger: logging.Logger, entry: SentimentJobLogEntry) -> None:
    if entry.status == "SUCCESS":
        logger.info(
            "Note %s labeled %s (%.3f)",
            entry.note_id,
            entry.label,
            entry.score or 0.0,
        )
    elif entry.status == "SKIPPED":
        logger.info("Note %s skipped: %s", entry.note_id, entry.message or "")
    else:
        logger.error("Note %s failed: %s", entry.note_id, entry.message or "")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = run_sentiment_job(logger=LOGGER)
    for entry in result.logs:
        _log_entry(LOGGER, entry)

    LOGGER.info(
        "Sentiment job complete: %d processed, %d skipped, %d failed",
        result.processed,
        result.skipped,
        result.failed,
    )


if __name__ == "__main__":
    main()

