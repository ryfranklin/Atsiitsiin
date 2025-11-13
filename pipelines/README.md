# Sentiment Pipeline

The Snowpark pipeline enriches notes with sentiment using Snowflake Cortex models. It performs three main tasks:

1. **Fetch** notes from `NOTES` where `SENTIMENT_ANALYZED_AT` is null.
2. **Invoke** `SNOWFLAKE.CORTEX.SENTIMENT` on each note text.
3. **Update** `NOTES` with `SENTIMENT_LABEL`, `SENTIMENT_SCORE`, `SENTIMENT_ANALYZED_AT` and logs outcomes into `NOTE_SENTIMENT_WORKLOG`.

## Prerequisites

- Snowflake account with Snowpark and Cortex enabled.
- The `NOTES` table as defined in `schemas.sql` (plus sentiment columns added by `pipelines/sentiment.sql`).
- Environment variables for Snowflake connectivity (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_ROLE`, etc.) or a Snowflake connection profile.

## Running Manually

1. Apply schema updates:
   ```sql
   USE DATABASE <your_db>;
   USE SCHEMA <your_schema>;
   -- run the script
   !source pipelines/sentiment.sql
   ```

2. Execute the pipeline from the repo root:
   ```bash
   make run-pipeline
   ```

   This calls `pipelines/sentiment.py`, which creates a Snowpark session using environment variables and processes up to 100 pending notes per run.

## Scheduling in Snowflake

Create a Snowflake stored procedure wrapping `pipelines/sentiment.py`, then schedule it via a Task:

```sql
CREATE OR REPLACE PROCEDURE pipelines.run_sentiment()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'handler'
AS
$$
from pipelines.sentiment import run

def handler(session):
    run()
    return "ok"
$$;

CREATE OR REPLACE TASK pipelines.note_sentiment_task
WAREHOUSE = <your_wh>
SCHEDULE = 'USING CRON 5 * * * * UTC'
AS
CALL pipelines.run_sentiment();

ALTER TASK pipelines.note_sentiment_task RESUME;
```

## Monitoring

- Inspect `NOTE_SENTIMENT_WORKLOG` for success/failure history.
- Use the FastAPI endpoint `GET /notes/analytics/sentiment` to view aggregate stats.

