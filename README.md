# Atsiitsʼiin – An Agentic Second Brain

Atsiitsʼiin (Navajo: "the head") is a personal knowledge agent that captures, organizes, and retrieves your ideas using LLMs and semantic memory on Snowflake. It follows agentic design and the GAME architecture (Goal, Abilities, Memory, Environment).

Overview

- Capture thoughts, notes, and inspirations
- Store them with LLM-generated summaries and tags
- Recall related ideas using semantic search (Snowflake VECTOR)
- Summarize your thinking over **time**
- Connect related entries for deeper insight

GAME Architecture

| Component | Description |
|----------|-------------|
| Goal | Support external thinking and long-term memory |
| Abilities | Capture, categorize, summarize, recall, and suggest |
| Memory | Embeddings + metadata stored in Snowflake |
| Environment | CLI (example script) |

Project Structure

```text
atsiitsiin/
├── README.md
├── pyproject.toml
├── src/
│   └── atsiitsiin/
│       ├── __init__.py          # Public package API
│       ├── agents/              # LLM-enabled agent interfaces
│       ├── config/              # Application configuration helpers
│       ├── memory/              # Embedding, ingestion, retrieval utilities
│       └── integrations/
│           └── snowflake/       # Snowflake configuration & connections
├── examples/
│   └── run_atsiitsiin.py        # Demo CLI workflow
├── tests/
│   └── test_memory_store.py
├── docs/                        # Architecture notes & planning (optional)
└── .env.example                 # Snowflake + LLM env vars template
```

## Getting Started

1. Configure Snowflake and environment

Create a database and schema in Snowflake, then apply `schemas.sql` in your target database/schema.

Create an environment file in the project root (recommended):

Option A (use your own): create `.env` and add keys.

Option B (use template):

```sh
cp .env.example .env
```

Edit `.env` with your values:

```text
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_WAREHOUSE=**COMPUTE_WH
SNOWFLAKE_DATABASE=MS3DM
SNOWFLAKE_SCHEMA=ATSIITSIIN
SNOWFLAKE_ROLE=ACCOUNTADMIN
# One of the two auth methods:
SNOWFLAKE_PRIVATE_KEY_PATH=/absolute/path/to/private_key.p8
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE=your_passphrase
# Or:
# SNOWFLAKE_PASSWORD=your_password

# LLM provider (LiteLLM)
OPENAI_API_KEY=your_openai_api_key
```

1) Configure Snowflake and environment

Create a database and schema in Snowflake, then apply `schemas.sql` in your target database/schema.

Create an environment file in the project root (recommended):

Option A (use your own): create `.env` and add keys.

Option B (use template):

```sh
cp .env.example .env
```

Edit `.env` with your values:

```text
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_WAREHOUSE=**COMPUTE_WH
SNOWFLAKE_DATABASE=MS3DM
SNOWFLAKE_SCHEMA=ATSIITSIIN
SNOWFLAKE_ROLE=ACCOUNTADMIN
# One of the two auth methods:
SNOWFLAKE_PRIVATE_KEY_PATH=/absolute/path/to/private_key.p8
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE=your_passphrase
# Or:
# SNOWFLAKE_PASSWORD=your_password

# LLM provider (LiteLLM)
OPENAI_API_KEY=your_openai_api_key
```

1. Install dependencies

Use your project’s existing environment and install any missing deps (e.g., `litellm`, `snowflake-snowpark-python` if not already present).

1. Run the demo

```text
python examples/run_atsiitsiin.py
```

1. Start the FastAPI backend

```text
make run-api
```

This launches the HTTP API at `http://127.0.0.1:8000`. The automatically generated docs live at `/docs`.

1. Launch the Streamlit prototype UI

```text
make run-streamlit
```

By default the UI points to `http://127.0.0.1:8000`. Override by setting `ATSIIITSIN_API_URL`.

## Tagging Pipeline

Once notes are written to Snowflake, run the tagging pipeline to classify them with LLM-generated tags:

```python
uv run python - <<'PY'
from atsiitsiin.config import AtsiiitsiinConfig, get_snowflake_connection
from atsiitsiin.memory.store import SnowflakeMemoryStore
from atsiitsiin.pipelines.tagging import assign_tags_to_notes

cfg = AtsiiitsiinConfig()
with get_snowflake_connection(cfg.snowflake_env_file).session_context() as session:
    store = SnowflakeMemoryStore(session, cfg.embedding_dim)
    note_ids = store.list_note_ids_missing_tags(limit=25)

if not note_ids:
    print("No untagged notes found.")
else:
    results = assign_tags_to_notes(note_ids, cfg)
    for res in results:
        print(res.note_id, [t.tag for t in res.tags])
PY
```

> Tip: The Streamlit Notes view provides a **Tag untagged notes** button that calls the new `/notes/analytics/tagging/run` API and classifies any notes missing tags.

Call `classify_note_tags(<note_id>, cfg)` if you want to re-tag a specific note on demand.

## Sentiment Pipeline

Local Snowpark sentiment enrichment can be executed once you have Snowflake credentials configured in environment variables (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_PRIVATE_KEY_PATH`, etc.).

```text
make run-pipeline
```

This calls the Snowpark script in `pipelines/sentiment.py`, updating `NOTES` with `SENTIMENT_LABEL`, `SENTIMENT_SCORE` and logging to `NOTE_SENTIMENT_WORKLOG`.

This will:

- Store a note via the agent (`remember:` command)
- Search memory (`search:` command) and answer with retrieved context

Tech Stack

| Component | Tool |
|----------|------|
| LLM Interface | LiteLLM (direct integration) |
| Embeddings | `text-embedding-3-small` (configurable) |
| Vector Search | Snowflake VECTOR + cosine similarity |
| Database | Snowflake |
| Interface | Simple demo script (CLI) |

License

MIT — free to use, modify, and build upon.
