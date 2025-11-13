# pyright: reportCallIssue=false
import json
import uuid
from collections.abc import Iterable
from typing import Any

from snowflake.snowpark import Session


class SnowflakeMemoryStore:
    def __init__(self, session: Session, embedding_dim: int = 1536):
        self.session = session
        self.embedding_dim = embedding_dim

    def upsert_note_with_chunks(
        self,
        user_id: str,
        source: str,
        title: str,
        chunks: list[
            tuple[int, str, list[float]]
        ],  # (chunk_index, text, embedding)
    ) -> str:
        note_id = str(uuid.uuid4())

        self.session.sql(  # nosec B608
            "INSERT INTO NOTES (ID, USER_ID, SOURCE, TITLE) "
            f"VALUES ('{note_id}', '{self._escape(user_id)}', '{self._escape(source)}', '{self._escape(title)}')"
        ).collect()  # pyright: ignore[reportCallIssue]

        if chunks:
            select_rows = []
            for idx, text, emb in chunks:
                chunk_id = str(uuid.uuid4())
                emb_json = ", ".join(str(x) for x in emb)
                select_rows.append(
                    "SELECT "
                    f"'{chunk_id}' AS ID, "
                    f"'{note_id}' AS NOTE_ID, "
                    f"{idx} AS CHUNK_INDEX, "
                    f"'{self._escape(text)}' AS TEXT, "
                    f"PARSE_JSON('[{emb_json}]') AS EMBEDDING"
                )

            sql = (
                "INSERT INTO NOTE_CHUNKS (ID, NOTE_ID, CHUNK_INDEX, TEXT, EMBEDDING) "
                + " "
                + " UNION ALL ".join(select_rows)
            )

            self.session.sql(sql).collect()

        return note_id

    def similarity_search(self, query_embedding: list[float], limit: int = 10):
        rows = self.session.sql(  # nosec B608
            "SELECT ID, NOTE_ID, CHUNK_INDEX, TEXT, EMBEDDING FROM NOTE_CHUNKS"
        ).collect()  # pyright: ignore[reportCallIssue]

        def cosine(a: list[float], b: list[float]) -> float:
            import math

            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        results: list[dict[str, Any]] = []
        for r in rows:
            row_dict = self._row_to_dict(r)
            emb = self._parse_embedding(row_dict.get("EMBEDDING"))
            score = 1 - cosine(emb, query_embedding)
            note_id_any = row_dict.get("NOTE_ID")
            if isinstance(note_id_any, (str, int)):
                tags = self.list_note_tags(str(note_id_any))
            else:
                tags = []
            results.append(
                {
                    "ID": row_dict.get("ID"),
                    "NOTE_ID": row_dict.get("NOTE_ID"),
                    "CHUNK_INDEX": row_dict.get("CHUNK_INDEX"),
                    "TEXT": row_dict.get("TEXT"),
                    "distance": score,
                    "tags": tags,
                }
            )

        results.sort(key=lambda x: x["distance"])  # ascending distance
        return results[:limit]

    def list_notes(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        sql = (
            "SELECT ID, USER_ID, SOURCE, TITLE, CREATED_AT, "
            "CONTEXT_LABEL, CONTEXT_CONFIDENCE, CONTEXT_ANALYZED_AT "
            "FROM NOTES "
            f"ORDER BY CREATED_AT DESC NULLS LAST LIMIT {limit} OFFSET {offset}"
        )
        rows = self.session.sql(sql).collect()  # pyright: ignore[reportCallIssue]  # nosec B608
        notes = [self._row_to_dict(row) for row in rows]
        if not notes:
            return []

        note_ids = [
            str(note_id)
            for note_id in (note.get("ID") for note in notes)
            if isinstance(note_id, (str, int))
        ]
        tags_by_note: dict[str, list[dict[str, Any]]] = {}
        if note_ids:
            placeholders = ", ".join(f"'{self._escape(nid)}'" for nid in note_ids)
            tag_rows = self.session.sql(  # nosec B608
                "SELECT nt.NOTE_ID, t.TAG, t.DISPLAY_NAME, t.DESCRIPTION, t.COLOR_HEX, "
                "nt.SOURCE, nt.CONFIDENCE, nt.CREATED_AT, nt.UPDATED_AT "
                "FROM NOTE_TAGS nt LEFT JOIN TAGS t ON nt.TAG = t.TAG "
                f"WHERE nt.NOTE_ID IN ({placeholders}) "
                "ORDER BY CASE WHEN nt.SOURCE = 'user' THEN 0 ELSE 1 END, nt.CREATED_AT DESC"
            ).collect()  # pyright: ignore[reportCallIssue]
            for row in tag_rows:
                row_dict = self._row_to_dict(row)
                note_id = row_dict.get("NOTE_ID")
                if isinstance(note_id, (str, int)):
                    key = str(note_id)
                    tags_by_note.setdefault(key, []).append(row_dict)

        for note in notes:
            note_id = note.get("ID")
            if isinstance(note_id, (str, int)):
                note["tags"] = tags_by_note.get(str(note_id), [])
            else:
                note["tags"] = []

        return notes

    def get_note(
        self, note_id: str, include_chunks: bool = False
    ) -> dict[str, Any] | None:
        safe_id = self._escape(note_id)
        rows = self.session.sql(  # nosec B608
            "SELECT ID, USER_ID, SOURCE, TITLE, CREATED_AT, "
            "CONTEXT_LABEL, CONTEXT_CONFIDENCE, CONTEXT_ANALYZED_AT "
            f"FROM NOTES WHERE ID = '{safe_id}'"
        ).collect()  # pyright: ignore[reportCallIssue]
        if not rows:
            return None
        note = self._row_to_dict(rows[0])

        note_tags = self.session.sql(  # nosec B608
            "SELECT t.TAG, t.DISPLAY_NAME, t.DESCRIPTION, t.COLOR_HEX, "
            "nt.SOURCE, nt.CONFIDENCE, nt.CREATED_AT, nt.UPDATED_AT "
            f"FROM NOTE_TAGS nt LEFT JOIN TAGS t ON nt.TAG = t.TAG "
            f"WHERE nt.NOTE_ID = '{safe_id}' "
            "ORDER BY CASE WHEN nt.SOURCE = 'user' THEN 0 ELSE 1 END, nt.CREATED_AT DESC"
        ).collect()  # pyright: ignore[reportCallIssue]

        chunk_rows = self.session.sql(  # nosec B608
            "SELECT ID, NOTE_ID, CHUNK_INDEX, TEXT, EMBEDDING "
            f"FROM NOTE_CHUNKS WHERE NOTE_ID = '{safe_id}' "
            "ORDER BY CHUNK_INDEX"
        ).collect()  # pyright: ignore[reportCallIssue]

        chunk_dicts = [
            self._row_to_dict(r)
            for r in chunk_rows
        ]

        if include_chunks:
            note["chunks"] = [
                {
                    "ID": row_dict.get("ID"),
                    "NOTE_ID": row_dict.get("NOTE_ID"),
                    "CHUNK_INDEX": row_dict.get("CHUNK_INDEX"),
                    "TEXT": row_dict.get("TEXT"),
                    "EMBEDDING": self._parse_embedding(row_dict.get("EMBEDDING")),
                }
                for row_dict in chunk_dicts
            ]
        else:
            note["chunks"] = []

        note["DESCRIPTION"] = "\n\n".join(
            row_dict.get("TEXT", "") for row_dict in chunk_dicts if row_dict.get("TEXT")
        )

        note["tags"] = [
            self._row_to_dict(row) for row in note_tags
        ]

        return note

    def count_notes(self) -> int:
        rows = self.session.sql(  # pyright: ignore[reportCallIssue]
            "SELECT COUNT(1) AS TOTAL FROM NOTES"
        ).collect()
        if not rows:
            return 0
        first = rows[0]
        if hasattr(first, "__getitem__"):
            try:
                return int(first["TOTAL"])  # type: ignore[index]
            except (KeyError, TypeError, ValueError):
                try:
                    return int(first[0])  # type: ignore[index]
                except (KeyError, TypeError, ValueError):
                    pass
        if hasattr(first, "as_dict"):
            return int(first.as_dict().get("TOTAL", 0))
        if hasattr(first, "asDict"):
            return int(first.asDict().get("TOTAL", 0))
        return 0

    def upsert_note_tags(
        self,
        note_id: str,
        tags: Iterable[dict[str, Any]],
        source: str = "llm_suggestion",
    ) -> None:
        tag_entries = list(tags)
        if not tag_entries:
            return

        for entry in tag_entries:
            tag_value = entry.get("tag")
            if not tag_value:
                continue
            tag = self._escape(tag_value.strip())
            display_name = entry.get("display_name") or tag_value
            description = entry.get("description")
            color_hex = entry.get("color_hex")
            entry_source_raw = entry.get("source") or source or "llm_suggestion"
            entry_source = str(entry_source_raw).strip() or "llm_suggestion"
            confidence = entry.get("confidence")
            if entry_source == "user" and confidence is None:
                confidence = 1.0

            self.session.sql(
                """
                MERGE INTO TAGS AS target
                USING (
                    SELECT
                        ? AS TAG,
                        ? AS DISPLAY_NAME,
                        ? AS DESCRIPTION,
                        ? AS COLOR_HEX
                ) AS source
                ON target.TAG = source.TAG
                WHEN MATCHED THEN UPDATE SET
                    target.DISPLAY_NAME = COALESCE(source.DISPLAY_NAME, target.DISPLAY_NAME),
                    target.DESCRIPTION = COALESCE(source.DESCRIPTION, target.DESCRIPTION),
                    target.COLOR_HEX = COALESCE(source.COLOR_HEX, target.COLOR_HEX),
                    target.UPDATED_AT = CURRENT_TIMESTAMP()
                WHEN NOT MATCHED THEN INSERT (TAG, DISPLAY_NAME, DESCRIPTION, COLOR_HEX)
                VALUES (source.TAG, source.DISPLAY_NAME, source.DESCRIPTION, source.COLOR_HEX)
                """,
                params=[tag_value, display_name, description, color_hex],
            ).collect()

            confidence_sql = "NULL" if confidence is None else str(float(confidence))

            self.session.sql(  # nosec B608
                f"""
                MERGE INTO NOTE_TAGS AS target
                USING (
                    SELECT
                        '{self._escape(note_id)}' AS NOTE_ID,
                        '{tag}' AS TAG,
                        '{self._escape(entry_source)}' AS SOURCE,
                        {confidence_sql} AS CONFIDENCE
                ) AS source
                ON target.NOTE_ID = source.NOTE_ID AND target.TAG = source.TAG
                WHEN MATCHED THEN UPDATE SET
                    SOURCE = CASE
                        WHEN target.SOURCE = 'user' AND source.SOURCE <> 'user' THEN target.SOURCE
                        ELSE source.SOURCE
                    END,
                    CONFIDENCE = CASE
                        WHEN target.SOURCE = 'user' AND source.SOURCE <> 'user' THEN target.CONFIDENCE
                        ELSE source.CONFIDENCE
                    END,
                    UPDATED_AT = CURRENT_TIMESTAMP()
                WHEN NOT MATCHED THEN INSERT (NOTE_ID, TAG, SOURCE, CONFIDENCE)
                VALUES (source.NOTE_ID, source.TAG, source.SOURCE, source.CONFIDENCE)
                """,
            ).collect()

    def record_tag_audit(
        self,
        note_id: str,
        tags: Iterable[dict[str, Any]],
        raw_response: Any,
        source: str = "llm_suggestion",
    ) -> str:
        audit_id = str(uuid.uuid4())
        tag_list = list(tags)
        self.session.sql(
            """
            INSERT INTO NOTE_TAG_AUDIT (AUDIT_ID, NOTE_ID, TAGS, RAW_RESPONSE, SOURCE)
            SELECT ?, ?, PARSE_JSON(?), PARSE_JSON(?), ?
            """,
            params=[
                audit_id,
                note_id,
                json.dumps(tag_list),
                json.dumps(raw_response),
                source,
            ],
        ).collect()
        return audit_id

    def list_note_tags(self, note_id: str) -> list[dict[str, Any]]:
        safe_id = self._escape(note_id)
        rows = self.session.sql(  # nosec B608
            "SELECT t.TAG, t.DISPLAY_NAME, t.DESCRIPTION, t.COLOR_HEX, "
            "nt.SOURCE, nt.CONFIDENCE, nt.CREATED_AT, nt.UPDATED_AT "
            f"FROM NOTE_TAGS nt LEFT JOIN TAGS t ON nt.TAG = t.TAG "
            f"WHERE nt.NOTE_ID = '{safe_id}' "
            "ORDER BY CASE WHEN nt.SOURCE = 'user' THEN 0 ELSE 1 END, nt.CREATED_AT DESC"
        ).collect()  # pyright: ignore[reportCallIssue]
        return [self._row_to_dict(row) for row in rows]

    def list_tags(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        rows = self.session.sql(  # nosec B608
            "SELECT t.TAG, t.DISPLAY_NAME, t.DESCRIPTION, t.COLOR_HEX, "
            "COUNT(nt.NOTE_ID) AS NOTE_COUNT, MAX(nt.CREATED_AT) AS LAST_USED_AT "
            "FROM TAGS t LEFT JOIN NOTE_TAGS nt ON t.TAG = nt.TAG "
            "GROUP BY t.TAG, t.DISPLAY_NAME, t.DESCRIPTION, t.COLOR_HEX "
            f"ORDER BY NOTE_COUNT DESC NULLS LAST LIMIT {limit} OFFSET {offset}"
        ).collect()  # pyright: ignore[reportCallIssue]
        return [self._row_to_dict(row) for row in rows]

    def list_note_ids_missing_tags(self, limit: int = 50) -> list[str]:
        limit = max(1, int(limit))
        rows = self.session.sql(  # nosec B608
            "SELECT n.ID FROM NOTES n "
            "LEFT JOIN NOTE_TAGS nt ON n.ID = nt.NOTE_ID "
            "GROUP BY n.ID "
            "HAVING COUNT(nt.TAG) = 0 "
            "ORDER BY MAX(n.CREATED_AT) DESC NULLS LAST "
            f"LIMIT {limit}"
        ).collect()  # pyright: ignore[reportCallIssue]
        note_ids: list[str] = []
        for row in rows:
            row_dict = self._row_to_dict(row)
            note_id = row_dict.get("ID")
            if isinstance(note_id, (str, int)):
                note_ids.append(str(note_id))
        return note_ids

    def delete_note(self, note_id: str) -> bool:
        safe_id = self._escape(note_id)
        existing = self.get_note(note_id)
        if existing is None:
            return False
        self.session.sql(  # nosec B608
            f"DELETE FROM NOTE_TAG_AUDIT WHERE NOTE_ID = '{safe_id}'"
        ).collect()
        self.session.sql(  # nosec B608
            f"DELETE FROM NOTE_TAGS WHERE NOTE_ID = '{safe_id}'"
        ).collect()
        self.session.sql(  # nosec B608
            f"DELETE FROM NOTE_CHUNKS WHERE NOTE_ID = '{safe_id}'"
        ).collect()
        self.session.sql(  # nosec B608
            f"DELETE FROM NOTES WHERE ID = '{safe_id}'"
        ).collect()
        return True

    def remove_note_tags(self, note_id: str, tags: Iterable[str]) -> list[dict[str, Any]]:
        tag_list = [tag for tag in (t.strip() for t in tags) if tag]
        if not tag_list:
            return self.list_note_tags(note_id)
        safe_id = self._escape(note_id)
        tag_placeholders = ", ".join(f"'{self._escape(tag)}'" for tag in tag_list)
        self.session.sql(  # nosec B608
            "DELETE FROM NOTE_TAGS "
            f"WHERE NOTE_ID = '{safe_id}' "
            f"AND TAG IN ({tag_placeholders})"
        ).collect()
        return self.list_note_tags(note_id)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _parse_embedding(value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = json.loads(value)
        else:
            raw = list(value)
        return [float(v) for v in raw]

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if hasattr(row, "as_dict"):
            return row.as_dict()  # type: ignore[return-value]
        if hasattr(row, "asDict"):
            return row.asDict()  # type: ignore[return-value]
        if hasattr(row, "keys"):
            return {key: row[key] for key in row.keys()}  # type: ignore[index]
        return dict(row)
