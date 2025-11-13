from __future__ import annotations

import logging
from collections.abc import Iterable

from atsiitsiin.config import AtsiiitsiinConfig, get_snowflake_connection

LOGGER = logging.getLogger("atsiitsiin.migrations.move_public_to_atsiitsiin")


def _qualify(database: str, schema: str, identifier: str) -> str:
    return f'{database}.{schema}.{identifier}'


def _table_exists(session, database: str, schema: str, table_name: str) -> bool:
    show_sql = (
        "SHOW TABLES LIKE %s IN SCHEMA %s"
    )
    result = session.sql(show_sql, params=[table_name, f"{database}.{schema}"]).collect()
    return bool(result)


def move_table(
    session,
    database: str,
    source_schema: str,
    target_schema: str,
    table_name: str,
) -> None:
    if not _table_exists(session, database, source_schema, table_name):
        LOGGER.info("Table %s.%s.%s not found; skipping.", database, source_schema, table_name)
        return

    qualified_source = _qualify(database, source_schema, table_name)
    qualified_target = _qualify(database, target_schema, table_name)
    LOGGER.info("Moving %s to %s", qualified_source, qualified_target)
    session.sql(
        f"ALTER TABLE {qualified_source} RENAME TO {qualified_target}"
    ).collect()


def apply_grants(session, database: str, schema: str, role: str) -> None:
    LOGGER.info("Applying grants on %s.%s for role %s", database, schema, role)
    session.sql(f"GRANT USAGE ON SCHEMA {database}.{schema} TO ROLE {role}").collect()
    session.sql(
        f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA {database}.{schema} TO ROLE {role}"
    ).collect()
    session.sql(
        f"GRANT ALL PRIVILEGES ON FUTURE TABLES IN SCHEMA {database}.{schema} TO ROLE {role}"
    ).collect()


def run(
    *,
    database: str = "MS3DM",
    source_schema: str = "PUBLIC",
    target_schema: str = "ATSIITSIIN",
    tables: Iterable[str] | None = None,
    grant_role: str = "MS3DM_ACCOUNT_ADMIN",
    cfg: AtsiiitsiinConfig | None = None,
) -> None:
    config = cfg or AtsiiitsiinConfig()
    tables_to_move = list(
        tables
        or [
            "NOTES",
            "NOTE_CHUNKS",
            "NOTE_CONTEXT_WORKLOG",
            "NOTE_SENTIMENT_WORKLOG",
            "TAGS",
            "NOTE_TAGS",
            "NOTE_TAG_AUDIT",
        ]
    )

    with get_snowflake_connection(config.snowflake_env_file).session_context() as session:
        LOGGER.info("Ensuring schema %s.%s exists", database, target_schema)
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{target_schema}").collect()

        for table in tables_to_move:
            move_table(session, database, source_schema, target_schema, table)

        apply_grants(session, database, target_schema, grant_role)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()

