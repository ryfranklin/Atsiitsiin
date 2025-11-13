from atsiitsiin.memory import SnowflakeMemoryStore


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def collect(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows=None):
        self.queries: list[str] = []
        self._rows = rows or []
        self._call_index = 0

    def sql(self, sql: str):
        self.queries.append(sql)
        if self._rows and isinstance(self._rows[0], list):
            index = min(self._call_index, len(self._rows) - 1)
            rows = self._rows[index]
        else:
            rows = self._rows
        self._call_index += 1
        return _FakeResult(rows)


def test_similarity_search_sorts_by_distance():
    rows = [
        {
            "ID": "c1",
            "NOTE_ID": "n1",
            "CHUNK_INDEX": 0,
            "TEXT": "a",
            "EMBEDDING": [1.0, 0.0],
        },
        {
            "ID": "c2",
            "NOTE_ID": "n1",
            "CHUNK_INDEX": 1,
            "TEXT": "b",
            "EMBEDDING": [0.0, 1.0],
        },
    ]
    sess = _FakeSession([rows, []])
    store = SnowflakeMemoryStore(sess, embedding_dim=2)  # type: ignore[arg-type]

    results = store.similarity_search([1.0, 0.0], limit=2)

    assert len(results) == 2
    assert results[0]["ID"] == "c1"  # distance 0.0 first
    assert results[1]["ID"] == "c2"
    assert results[0]["tags"] == []
    assert results[1]["tags"] == []


def test_upsert_note_generates_expected_sql():
    sess = _FakeSession()
    store = SnowflakeMemoryStore(sess, embedding_dim=2)  # type: ignore[arg-type]

    note_id = store.upsert_note_with_chunks(
        user_id="u1",
        source="manual",
        title="t1",
        chunks=[(0, "hello", [0.1, 0.2])],
    )

    assert any("INSERT INTO NOTES" in q for q in sess.queries)
    assert any("INSERT INTO NOTE_CHUNKS" in q for q in sess.queries)
    assert any("PARSE_JSON" in q for q in sess.queries)
    assert isinstance(note_id, str)


def test_list_notes_returns_expected_rows():
    rows = [
        {
            "ID": "n1",
            "USER_ID": "u1",
            "SOURCE": "manual",
            "TITLE": "First note",
            "CREATED_AT": None,
        }
    ]
    sess = _FakeSession([rows, []])
    store = SnowflakeMemoryStore(sess, embedding_dim=2)  # type: ignore[arg-type]

    result = store.list_notes(limit=5, offset=10)
    assert result == [{**rows[0], "tags": []}]
    query = sess.queries[0]
    assert "LIMIT 5" in query
    assert "OFFSET 10" in query


def test_get_note_returns_chunks_when_requested():
    note_row = [
        {
            "ID": "note-1",
            "USER_ID": "user-1",
            "SOURCE": "manual",
            "TITLE": "Example",
            "CREATED_AT": None,
        }
    ]
    chunk_row = [
        {
            "ID": "chunk-1",
            "NOTE_ID": "note-1",
            "CHUNK_INDEX": 0,
            "TEXT": "hello",
            "EMBEDDING": "[0.1, 0.2]",
        }
    ]
    sess = _FakeSession([note_row, [], chunk_row])
    store = SnowflakeMemoryStore(sess, embedding_dim=2)  # type: ignore[arg-type]

    note = store.get_note("note-1", include_chunks=True)
    assert note is not None
    assert note["ID"] == "note-1"
    assert note["tags"] == []
    assert "chunks" in note
    assert note["chunks"][0]["TEXT"] == "hello"
    assert note["chunks"][0]["EMBEDDING"] == [0.1, 0.2]


def test_count_notes_executes_expected_query():
    sess = _FakeSession([{"TOTAL": 3}])
    store = SnowflakeMemoryStore(sess, embedding_dim=2)  # type: ignore[arg-type]

    total = store.count_notes()
    assert total == 3
    assert any("COUNT" in q.upper() for q in sess.queries)
