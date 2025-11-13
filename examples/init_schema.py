from pathlib import Path

from atsiitsiin.config import get_snowflake_connection


def main() -> None:
    conn = get_snowflake_connection()

    sql_path = Path("schemas.sql")
    sql_text = sql_path.read_text(encoding="utf-8")
    # Execute all statements in the schema file
    for raw in sql_text.split(";"):
        # Strip whitespace and drop comment-only lines
        lines = [
            ln for ln in raw.splitlines() if not ln.strip().startswith("--")
        ]
        stmt = "\n".join(lines).strip()
        if not stmt:
            continue
        conn.execute_sql(stmt)  # pyright: ignore[reportCallIssue]

    print("Schema initialized.")


if __name__ == "__main__":
    main()
