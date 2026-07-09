"""Postgres/pgvector storage helpers for Vercel Marketplace databases.

The local backend remains the default. When ``STORAGE_BACKEND=postgres`` this
module provides the durable stores that replace the markdown/sqlite files under
``memory/`` with Neon Postgres tables, while preserving the public store/index
APIs used by the rest of the assistant.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import Settings
from .memory.store import INDEX_FILENAME, Note


def enabled(settings: Settings) -> bool:
    return settings.storage_backend == "postgres"


def require_url(settings: Settings) -> str:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
    return settings.database_url


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised in deployed config
        raise RuntimeError(
            "STORAGE_BACKEND=postgres requires the 'psycopg[binary]' dependency"
        ) from exc
    return psycopg


@contextmanager
def connect(settings: Settings):
    psycopg = _psycopg()
    conn = psycopg.connect(require_url(settings), autocommit=False)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.9g}" for v in vector) + "]"


def _rows(cur) -> list[dict]:
    cols = [col.name for col in cur.description or []]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _executemany(conn, sql: str, params: list) -> None:
    if not params:
        return
    with conn.cursor() as cur:
        cur.executemany(sql, params)


def ensure_memory_schema(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_memory_notes (
              name TEXT PRIMARY KEY,
              description TEXT NOT NULL DEFAULT '',
              body TEXT NOT NULL DEFAULT '',
              kind TEXT NOT NULL DEFAULT 'semantic',
              salience DOUBLE PRECISION NOT NULL DEFAULT 0.5,
              confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
              tags JSONB NOT NULL DEFAULT '[]'::jsonb,
              source TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT '',
              last_recalled TEXT NOT NULL DEFAULT '',
              recall_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_memory_index (
              name TEXT PRIMARY KEY REFERENCES assistant_memory_notes(name)
                ON DELETE CASCADE,
              description TEXT NOT NULL DEFAULT '',
              kind TEXT NOT NULL DEFAULT 'semantic',
              salience DOUBLE PRECISION NOT NULL DEFAULT 0.5,
              updated TEXT NOT NULL DEFAULT '',
              last_recalled TEXT NOT NULL DEFAULT '',
              recall_count INTEGER NOT NULL DEFAULT 0,
              hash TEXT NOT NULL DEFAULT '',
              embedding vector NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_memory_trash (
              id BIGSERIAL PRIMARY KEY,
              deleted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              name TEXT NOT NULL,
              description TEXT NOT NULL DEFAULT '',
              body TEXT NOT NULL DEFAULT '',
              kind TEXT NOT NULL DEFAULT 'semantic',
              salience DOUBLE PRECISION NOT NULL DEFAULT 0.5,
              confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
              tags JSONB NOT NULL DEFAULT '[]'::jsonb,
              source TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT '',
              last_recalled TEXT NOT NULL DEFAULT '',
              recall_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS assistant_memory_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )


def _note_from_row(row: dict) -> Note:
    tags = row.get("tags") or []
    return Note(
        name=str(row["name"]),
        description=str(row.get("description") or ""),
        body=str(row.get("body") or ""),
        kind=str(row.get("kind") or "semantic"),
        salience=float(row.get("salience") or 0.5),
        confidence=float(row.get("confidence") or 0.8),
        tags=list(tags),
        source=str(row.get("source") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
        last_recalled=str(row.get("last_recalled") or ""),
        recall_count=int(row.get("recall_count") or 0),
    )


def write_note(settings: Settings, note: Note) -> Path:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_memory_notes
              (name, description, body, kind, salience, confidence, tags, source,
               created, updated, last_recalled, recall_count)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
            ON CONFLICT(name) DO UPDATE SET
              description = excluded.description,
              body = excluded.body,
              kind = excluded.kind,
              salience = excluded.salience,
              confidence = excluded.confidence,
              tags = excluded.tags,
              source = excluded.source,
              created = excluded.created,
              updated = excluded.updated,
              last_recalled = excluded.last_recalled,
              recall_count = excluded.recall_count
            """,
            (
                note.name,
                note.description,
                note.body,
                note.kind,
                note.salience,
                note.confidence,
                __import__('json').dumps(note.tags),
                note.source,
                note.created,
                note.updated,
                note.last_recalled,
                note.recall_count,
            ),
        )
    return virtual_note_path(settings, note)


def virtual_note_path(settings: Settings, note: Note) -> Path:
    return settings.memory_path / ".postgres" / note.kind / f"{note.name}.md"


def list_notes(settings: Settings) -> list[Note]:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT name, description, body, kind, salience, confidence, tags,"
            " source, created, updated, last_recalled, recall_count"
            " FROM assistant_memory_notes ORDER BY name"
        )
        return [_note_from_row(row) for row in _rows(cur)]


def find_note(settings: Settings, name: str) -> Note | None:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT name, description, body, kind, salience, confidence, tags,"
            " source, created, updated, last_recalled, recall_count"
            " FROM assistant_memory_notes WHERE name = %s",
            (name,),
        )
        rows = _rows(cur)
    return _note_from_row(rows[0]) if rows else None


def unique_name(settings: Settings, slug: str, keep: str | None = None) -> str:
    existing = {n.name for n in list_notes(settings)}
    if slug == keep or slug not in existing:
        return slug
    i = 2
    while f"{slug}-{i}" in existing:
        i += 1
    return f"{slug}-{i}"


def purge_stale_files(settings: Settings, name: str, keep_kind: str) -> None:
    # Postgres stores one row per memory name, so there are no stale kind files.
    return None


def delete_note(settings: Settings, name: str) -> Note | None:
    note = find_note(settings, name)
    if note is None:
        return None
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_memory_trash
              (name, description, body, kind, salience, confidence, tags, source,
               created, updated, last_recalled, recall_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
            """,
            (
                note.name,
                note.description,
                note.body,
                note.kind,
                note.salience,
                note.confidence,
                __import__('json').dumps(note.tags),
                note.source,
                note.created,
                note.updated,
                note.last_recalled,
                note.recall_count,
            ),
        )
        conn.execute("DELETE FROM assistant_memory_notes WHERE name = %s", (name,))
    return note


def prune_trash(settings: Settings, max_age_days: int) -> int:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "DELETE FROM assistant_memory_trash "
            "WHERE deleted_at <= now() - (%s::text || ' days')::interval "
            "RETURNING id",
            (int(max_age_days),),
        )
        return len(cur.fetchall())


def regenerate_index(settings: Settings) -> Path:
    notes = list_notes(settings)
    lines = ["# Memory index", ""]
    if not notes:
        lines.append("_(empty)_")
    else:
        by_kind: dict[str, list[Note]] = {}
        for note in notes:
            by_kind.setdefault(note.kind, []).append(note)
        order = ["semantic", "procedural", "episodic"] + [
            k for k in by_kind if k not in {"semantic", "procedural", "episodic"}
        ]
        for kind in order:
            group = by_kind.get(kind)
            if not group:
                continue
            lines.append(f"## {kind.capitalize()}")
            for note in group:
                lines.append(f"- **{note.name}** — {note.description}")
            lines.append("")
    path = settings.memory_path / INDEX_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def read_index(settings: Settings) -> str:
    return regenerate_index(settings).read_text(encoding="utf-8").strip()


def _meta_get(conn, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM assistant_memory_meta WHERE key = %s", (key,)
    ).fetchone()
    return row[0] if row else None


def _meta_set(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO assistant_memory_meta(key, value) VALUES (%s, %s) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def upsert_memory_index(
    settings: Settings,
    name: str,
    path: str,
    description: str,
    vector: list[float],
    *,
    kind: str = "semantic",
    salience: float = 0.5,
    updated: str = "",
    last_recalled: str = "",
    recall_count: int = 0,
    text_hash: str = "",
) -> None:
    del path  # path is meaningful only for the local markdown backend.
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        dim = _meta_get(conn, "dim")
        if dim is None:
            _meta_set(conn, "dim", str(len(vector)))
            _meta_set(conn, "embedding_model", settings.embedding_model)
        conn.execute(
            """
            INSERT INTO assistant_memory_index
              (name, description, kind, salience, updated, last_recalled,
               recall_count, hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT(name) DO UPDATE SET
              description = excluded.description,
              kind = excluded.kind,
              salience = excluded.salience,
              updated = excluded.updated,
              last_recalled = excluded.last_recalled,
              recall_count = excluded.recall_count,
              hash = excluded.hash,
              embedding = excluded.embedding
            """,
            (
                name,
                description,
                kind,
                float(salience),
                updated,
                last_recalled,
                int(recall_count),
                text_hash,
                vector_literal(vector),
            ),
        )


def remove_memory_index(settings: Settings, name: str) -> bool:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "DELETE FROM assistant_memory_index WHERE name = %s RETURNING name", (name,)
        )
        return cur.fetchone() is not None


def search_memory_index(
    settings: Settings, query_vector: list[float], k: int
) -> list[tuple[str, str, str, str, float, int, str, float]]:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            """
            SELECT i.name, i.description, i.kind, i.salience, i.recall_count,
                   i.last_recalled, 1 - (i.embedding <=> %s::vector) AS similarity
            FROM assistant_memory_index i
            JOIN assistant_memory_notes n ON n.name = i.name
            ORDER BY i.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(query_vector), vector_literal(query_vector), int(k)),
        )
        rows = _rows(cur)
    return [
        (
            str(r["name"]),
            str(virtual_note_path(settings, Note(name=str(r["name"]), description="", body=""))),
            str(r["description"]),
            str(r["kind"]),
            float(r["salience"]),
            int(r["recall_count"]),
            str(r["last_recalled"] or ""),
            float(r["similarity"]),
        )
        for r in rows
    ]


def list_memory_entries(settings: Settings) -> list[tuple[str, str, str, float, int, str, str]]:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT name, description, kind, salience, recall_count, last_recalled,"
            " updated FROM assistant_memory_index"
        )
        rows = _rows(cur)
    return [
        (
            str(r["name"]),
            str(r["description"]),
            str(r["kind"]),
            float(r["salience"]),
            int(r["recall_count"]),
            str(r["last_recalled"] or ""),
            str(r["updated"] or ""),
        )
        for r in rows
    ]


def bump_turn_counter(settings: Settings) -> int:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_memory_meta(key, value) VALUES ('turn_count', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = "
            "(assistant_memory_meta.value::integer + 1)::text"
        )
        row = conn.execute(
            "SELECT value FROM assistant_memory_meta WHERE key = 'turn_count'"
        ).fetchone()
        return int(row[0])


def bump_recall(settings: Settings, names: list[str]) -> None:
    if not names:
        return
    today = datetime.now(timezone.utc).date().isoformat()
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        _executemany(
            conn,
            "UPDATE assistant_memory_index "
            "SET recall_count = recall_count + 1, last_recalled = %s "
            "WHERE name = %s",
            [(today, name) for name in names],
        )


def set_salience(settings: Settings, name: str, salience: float) -> None:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "UPDATE assistant_memory_index SET salience = %s WHERE name = %s",
            (float(salience), name),
        )


def get_stats(settings: Settings, name: str) -> tuple[int, str] | None:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT recall_count, last_recalled FROM assistant_memory_index WHERE name = %s",
            (name,),
        ).fetchone()
    return (int(row[0]), str(row[1] or "")) if row else None


def reindex_memory(settings: Settings) -> int:
    from .memory import index
    from .memory.embeddings import embed_passages

    notes = list_notes(settings)
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        stored_model = _meta_get(conn, "embedding_model")
        model_changed = stored_model not in (None, settings.embedding_model)
        prior = {
            name: (int(rc), str(lr or ""))
            for name, rc, lr in conn.execute(
                "SELECT name, recall_count, last_recalled FROM assistant_memory_index"
            ).fetchall()
        }
        if model_changed:
            conn.execute("DELETE FROM assistant_memory_index")
            conn.execute("DELETE FROM assistant_memory_meta WHERE key IN ('dim', 'embedding_model')")

    pending: list[tuple[Note, str, int, str]] = []
    live_names = {note.name for note in notes}
    with connect(settings) as conn:
        for note in notes:
            text_hash = index.content_hash(note.index_text)
            rc, lr = prior.get(note.name, (note.recall_count, note.last_recalled))
            row = conn.execute(
                "SELECT hash FROM assistant_memory_index WHERE name = %s", (note.name,)
            ).fetchone()
            if not model_changed and row is not None and row[0] == text_hash:
                conn.execute(
                    "UPDATE assistant_memory_index SET description = %s, kind = %s,"
                    " updated = %s WHERE name = %s",
                    (note.description, note.kind, note.updated, note.name),
                )
                continue
            pending.append((note, text_hash, rc, lr))
        stale = [
            row[0]
            for row in conn.execute("SELECT name FROM assistant_memory_index").fetchall()
            if row[0] not in live_names
        ]
        for name in stale:
            conn.execute("DELETE FROM assistant_memory_index WHERE name = %s", (name,))

    vectors = embed_passages([n.index_text for n, _h, _rc, _lr in pending], settings) if pending else []
    if len(vectors) != len(pending):
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(pending)} notes")
    for (note, text_hash, rc, lr), vector in zip(pending, vectors):
        upsert_memory_index(
            settings,
            note.name,
            str(virtual_note_path(settings, note)),
            note.description,
            vector,
            kind=note.kind,
            salience=note.salience,
            updated=note.updated,
            last_recalled=lr,
            recall_count=rc,
            text_hash=text_hash,
        )
    return len(live_names)


def ensure_docs_schema(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_documents (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              text TEXT NOT NULL,
              added TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_document_chunks (
              id BIGSERIAL PRIMARY KEY,
              doc_id TEXT NOT NULL REFERENCES assistant_documents(id) ON DELETE CASCADE,
              ord INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding vector NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS assistant_docs_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )


def docs_meta_get(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM assistant_docs_meta WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def docs_meta_set(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO assistant_docs_meta(key, value) VALUES (%s, %s) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )



def reindex_docs(settings: Settings) -> int:
    """Rebuild document chunk vectors from the stored text when the model changed.

    The docs mirror of :func:`reindex_memory`. ``assistant_document_chunks.embedding``
    is an undimensioned ``vector``, so a swapped embedding model does not fail on
    insert — it quietly mixes dimensions, and the ``<=>`` cosine operator then raises
    at query time. Rebuilding from ``assistant_documents.text`` (this store's source
    of truth) keeps every chunk on one model. No-op when the model is unchanged.
    """
    from .docs.store import chunk_text
    from .memory.embeddings import embed_passages

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        stored_model = docs_meta_get(conn, "embedding_model")
        rows = _rows(conn.execute("SELECT id, text FROM assistant_documents"))
        model_changed = stored_model != settings.embedding_model
        if not rows or not model_changed:
            return len(rows)
        conn.execute("DELETE FROM assistant_document_chunks")
        conn.execute(
            "DELETE FROM assistant_docs_meta WHERE key IN ('dim', 'embedding_model')"
        )

    for row in rows:
        pieces = chunk_text(str(row["text"]), settings.docs_chunk_chars)
        vectors = embed_passages(pieces, settings) if pieces else []
        if len(vectors) != len(pieces):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(pieces)} chunks"
            )
        with connect(settings) as conn:
            if vectors and docs_meta_get(conn, "dim") is None:
                docs_meta_set(conn, "dim", str(len(vectors[0])))
                docs_meta_set(conn, "embedding_model", settings.embedding_model)
            for ord_, (piece, vector) in enumerate(zip(pieces, vectors)):
                conn.execute(
                    "INSERT INTO assistant_document_chunks(doc_id, ord, text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (str(row["id"]), ord_, piece, vector_literal(vector)),
                )

    # A corpus whose documents are all empty never records a dim above; stamp the
    # model anyway so the next startup doesn't see a "changed" model and rebuild.
    with connect(settings) as conn:
        docs_meta_set(conn, "embedding_model", settings.embedding_model)
    return len(rows)


def add_document(settings: Settings, title: str, text: str, pieces: list[str], vectors: list[list[float]]):
    from .docs.store import Document
    import uuid

    ensure_docs_schema(settings)
    doc = Document(
        id=uuid.uuid4().hex[:12],
        title=title.strip() or "(untitled)",
        text=text,
        added=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        chunks=len(pieces),
    )
    with connect(settings) as conn:
        if vectors:
            if docs_meta_get(conn, "dim") is None:
                docs_meta_set(conn, "dim", str(len(vectors[0])))
                docs_meta_set(conn, "embedding_model", settings.embedding_model)
        conn.execute(
            "INSERT INTO assistant_documents(id, title, text, added) VALUES (%s, %s, %s, %s)",
            (doc.id, doc.title, doc.text, doc.added),
        )
        for ord_, (piece, vector) in enumerate(zip(pieces, vectors)):
            conn.execute(
                "INSERT INTO assistant_document_chunks(doc_id, ord, text, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                (doc.id, ord_, piece, vector_literal(vector)),
            )
    return doc


def list_documents(settings: Settings):
    from .docs.store import Document

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT d.id, d.title, d.text, d.added, COUNT(c.id) AS chunks "
            "FROM assistant_documents d "
            "LEFT JOIN assistant_document_chunks c ON c.doc_id = d.id "
            "GROUP BY d.id, d.title, d.text, d.added "
            "ORDER BY d.added DESC"
        )
        rows = _rows(cur)
    return [
        Document(
            id=str(r["id"]), title=str(r["title"]), text=str(r["text"]),
            added=str(r["added"]), chunks=int(r["chunks"] or 0),
        )
        for r in rows
    ]


def get_document(settings: Settings, doc_id: str):
    from .docs.store import Document

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT id, title, text, added FROM assistant_documents WHERE id = %s", (doc_id,)
        )
        rows = _rows(cur)
    if not rows:
        return None
    r = rows[0]
    return Document(id=str(r["id"]), title=str(r["title"]), text=str(r["text"]), added=str(r["added"]))


def delete_document(settings: Settings, doc_id: str) -> bool:
    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "DELETE FROM assistant_documents WHERE id = %s RETURNING id", (doc_id,)
        )
        return cur.fetchone() is not None


def search_chunks(settings: Settings, query_vector: list[float], top_k: int):
    from .docs.store import Chunk

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            """
            SELECT c.doc_id, c.text, d.title,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM assistant_document_chunks c
            JOIN assistant_documents d ON d.id = c.doc_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(query_vector), vector_literal(query_vector), int(top_k)),
        )
        rows = _rows(cur)
    return [
        Chunk(
            doc_id=str(r["doc_id"]), doc_title=str(r["title"]),
            text=str(r["text"]), similarity=float(r["similarity"]),
        )
        for r in rows
        if float(r["similarity"]) >= settings.docs_min_similarity
    ]


def reindex_documents(settings: Settings, chunker, embedder) -> int:
    ensure_docs_schema(settings)
    with connect(settings) as conn:
        stored_model = docs_meta_get(conn, "embedding_model")
        documents = _rows(conn.execute("SELECT id, text FROM assistant_documents"))
        if not documents or stored_model == settings.embedding_model:
            return len(documents)
        conn.execute("DELETE FROM assistant_document_chunks")
        conn.execute("DELETE FROM assistant_docs_meta WHERE key IN ('dim', 'embedding_model')")

    for doc in documents:
        pieces = chunker(str(doc["text"]), settings.docs_chunk_chars)
        vectors = embedder(pieces, settings) if pieces else []
        if len(vectors) != len(pieces):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(pieces)} chunks"
            )
        with connect(settings) as conn:
            if vectors and docs_meta_get(conn, "dim") is None:
                docs_meta_set(conn, "dim", str(len(vectors[0])))
                docs_meta_set(conn, "embedding_model", settings.embedding_model)
            for ord_, (piece, vector) in enumerate(zip(pieces, vectors)):
                conn.execute(
                    "INSERT INTO assistant_document_chunks(doc_id, ord, text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (str(doc["id"]), ord_, piece, vector_literal(vector)),
                )
    return len(documents)



def ensure_calendar_schema(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_calendar_events (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              start TEXT NOT NULL,
              "end" TEXT NOT NULL DEFAULT '',
              location TEXT NOT NULL DEFAULT '',
              notes TEXT NOT NULL DEFAULT '',
              rrule TEXT NOT NULL DEFAULT '',
              exdates TEXT NOT NULL DEFAULT '',
              overrides TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_calendar_write_log (
              id BIGSERIAL PRIMARY KEY,
              thread_id TEXT NOT NULL,
              batch_id TEXT NOT NULL,
              event_id TEXT NOT NULL,
              op TEXT NOT NULL,
              summary TEXT NOT NULL,
              before_json TEXT,
              applied_at TEXT NOT NULL,
              undone_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_calendar_reminders_fired (
              event_id TEXT NOT NULL,
              event_start TEXT NOT NULL,
              lead_minutes INTEGER NOT NULL,
              fired_at TEXT NOT NULL,
              PRIMARY KEY (event_id, event_start, lead_minutes)
            )
            """
        )


def _event_from_row(row: dict):
    from .calendar.store import Event

    return Event(
        id=str(row["id"]),
        title=str(row["title"]),
        start=str(row["start"]),
        end=str(row.get("end") or ""),
        location=str(row.get("location") or ""),
        notes=str(row.get("notes") or ""),
        rrule=str(row.get("rrule") or ""),
        exdates=str(row.get("exdates") or ""),
        overrides=str(row.get("overrides") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
    )


def create_event(settings: Settings, title: str, start: str, end: str = "", location: str = "", notes: str = "", rrule: str = ""):
    from .calendar import store as calendar_store
    import uuid

    ensure_calendar_schema(settings)
    now = calendar_store._stamp_now(settings)
    event = calendar_store.Event(
        id=uuid.uuid4().hex[:12],
        title=title.strip(),
        start=calendar_store._normalize_stamp(settings, start),
        end=calendar_store._normalize_stamp(settings, end),
        location=location.strip(),
        notes=notes.strip(),
        rrule=rrule.strip(),
        created=now,
        updated=now,
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_calendar_events "
            "(id, title, start, \"end\", location, notes, rrule, created, updated) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (event.id, event.title, event.start, event.end, event.location, event.notes, event.rrule, event.created, event.updated),
        )
    return event


def get_event(settings: Settings, event_id: str):
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, start, \"end\", location, notes, rrule, exdates, overrides, created, updated FROM assistant_calendar_events WHERE id = %s", (event_id,)))
    return _event_from_row(rows[0]) if rows else None


def list_events(settings: Settings):
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, start, \"end\", location, notes, rrule, exdates, overrides, created, updated FROM assistant_calendar_events"))
    return [_event_from_row(r) for r in rows]


def update_event(settings: Settings, event_id: str, fields: dict[str, str]):
    from .calendar import store as calendar_store

    ensure_calendar_schema(settings)
    existing = get_event(settings, event_id)
    if existing is None:
        return None
    updates = {k: str(v).strip() for k, v in fields.items() if v is not None}
    for key in ("start", "end"):
        if key in updates:
            updates[key] = calendar_store._normalize_stamp(settings, updates[key])
    if not updates:
        return existing
    updates["updated"] = calendar_store._stamp_now(settings)
    column_map = {"end": "\"end\""}
    assignments = ", ".join(f"{column_map.get(k, k)} = %s" for k in updates)
    with connect(settings) as conn:
        conn.execute(
            f"UPDATE assistant_calendar_events SET {assignments} WHERE id = %s",
            (*updates.values(), event_id),
        )
    return get_event(settings, event_id)


def restore_event(settings: Settings, event) -> object:
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_calendar_events
              (id, title, start, "end", location, notes, rrule, exdates, overrides, created, updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id) DO UPDATE SET
              title = excluded.title,
              start = excluded.start,
              "end" = excluded."end",
              location = excluded.location,
              notes = excluded.notes,
              rrule = excluded.rrule,
              exdates = excluded.exdates,
              overrides = excluded.overrides,
              created = excluded.created,
              updated = excluded.updated
            """,
            (event.id, event.title, event.start, event.end, event.location, event.notes, event.rrule, event.exdates, event.overrides, event.created, event.updated),
        )
    return event


def delete_event(settings: Settings, event_id: str):
    existing = get_event(settings, event_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_calendar_events WHERE id = %s", (event_id,))
    return existing


def ensure_tasks_schema(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_tasks (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              done BOOLEAN NOT NULL DEFAULT FALSE,
              due TEXT NOT NULL DEFAULT '',
              notes TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT '',
              done_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_task_write_log (
              id BIGSERIAL PRIMARY KEY,
              thread_id TEXT NOT NULL,
              batch_id TEXT NOT NULL,
              task_id TEXT NOT NULL,
              op TEXT NOT NULL,
              summary TEXT NOT NULL,
              before_json TEXT,
              applied_at TEXT NOT NULL,
              undone_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_task_reminders_fired (
              task_id TEXT NOT NULL,
              due TEXT NOT NULL,
              lead_minutes INTEGER NOT NULL,
              fired_at TEXT NOT NULL,
              PRIMARY KEY (task_id, due, lead_minutes)
            )
            """
        )


def _task_from_row(row: dict):
    from .tasks.store import Task

    return Task(
        id=str(row["id"]),
        title=str(row["title"]),
        done=bool(row.get("done")),
        due=str(row.get("due") or ""),
        notes=str(row.get("notes") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
        done_at=str(row.get("done_at") or ""),
    )


def create_task(settings: Settings, title: str, due: str = "", notes: str = ""):
    from .tasks import store as task_store
    import uuid

    ensure_tasks_schema(settings)
    now = task_store._stamp_now(settings)
    task = task_store.Task(
        id=uuid.uuid4().hex[:12],
        title=title.strip(),
        done=False,
        due=task_store._normalize_due(settings, due),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_tasks (id, title, done, due, notes, created, updated, done_at) "
            "VALUES (%s, %s, FALSE, %s, %s, %s, %s, '')",
            (task.id, task.title, task.due, task.notes, task.created, task.updated),
        )
    return task


def get_task(settings: Settings, task_id: str):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, done, due, notes, created, updated, done_at FROM assistant_tasks WHERE id = %s", (task_id,)))
    return _task_from_row(rows[0]) if rows else None


def list_tasks(settings: Settings):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, done, due, notes, created, updated, done_at FROM assistant_tasks"))
    return [_task_from_row(r) for r in rows]


def update_task(settings: Settings, task_id: str, fields: dict[str, str]):
    from .tasks import store as task_store

    ensure_tasks_schema(settings)
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    updates = {k: str(v).strip() for k, v in fields.items() if v is not None}
    if "due" in updates:
        updates["due"] = task_store._normalize_due(settings, updates["due"])
    if not updates:
        return existing
    updates["updated"] = task_store._stamp_now(settings)
    assignments = ", ".join(f"{k} = %s" for k in updates)
    with connect(settings) as conn:
        conn.execute(f"UPDATE assistant_tasks SET {assignments} WHERE id = %s", (*updates.values(), task_id))
    return get_task(settings, task_id)


def complete_task(settings: Settings, task_id: str):
    from .tasks import store as task_store

    existing = get_task(settings, task_id)
    if existing is None or existing.done:
        return existing
    now = task_store._stamp_now(settings)
    with connect(settings) as conn:
        conn.execute("UPDATE assistant_tasks SET done = TRUE, done_at = %s, updated = %s WHERE id = %s", (now, now, task_id))
    return get_task(settings, task_id)


def restore_task(settings: Settings, task) -> object:
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_tasks (id, title, done, due, notes, created, updated, done_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id) DO UPDATE SET
              title = excluded.title,
              done = excluded.done,
              due = excluded.due,
              notes = excluded.notes,
              created = excluded.created,
              updated = excluded.updated,
              done_at = excluded.done_at
            """,
            (task.id, task.title, bool(task.done), task.due, task.notes, task.created, task.updated, task.done_at),
        )
    return task


def delete_task(settings: Settings, task_id: str):
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_tasks WHERE id = %s", (task_id,))
    return existing


def record_calendar_write(settings: Settings, thread_id: str, batch_id: str, event_id: str, op: str, summary: str, before_json: str | None, applied_at: str) -> None:
    if not thread_id:
        return
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_calendar_write_log (thread_id, batch_id, event_id, op, summary, before_json, applied_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (thread_id, batch_id, event_id, op, summary, before_json, applied_at),
        )


def calendar_write_rows(settings: Settings, thread_id: str) -> list[dict]:
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        return _rows(conn.execute("SELECT * FROM assistant_calendar_write_log WHERE thread_id = %s AND undone_at IS NULL ORDER BY id DESC", (thread_id,)))


def mark_calendar_writes_undone(settings: Settings, ids: list[int], undone_at: str) -> None:
    if not ids:
        return
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        _executemany(conn, "UPDATE assistant_calendar_write_log SET undone_at = %s WHERE id = %s", [(undone_at, i) for i in ids])


def record_task_write(settings: Settings, thread_id: str, batch_id: str, task_id: str, op: str, summary: str, before_json: str | None, applied_at: str) -> None:
    if not thread_id:
        return
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_task_write_log (thread_id, batch_id, task_id, op, summary, before_json, applied_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (thread_id, batch_id, task_id, op, summary, before_json, applied_at),
        )


def task_write_rows(settings: Settings, thread_id: str) -> list[dict]:
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        return _rows(conn.execute("SELECT * FROM assistant_task_write_log WHERE thread_id = %s AND undone_at IS NULL ORDER BY id DESC", (thread_id,)))


def mark_task_writes_undone(settings: Settings, ids: list[int], undone_at: str) -> None:
    if not ids:
        return
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        _executemany(conn, "UPDATE assistant_task_write_log SET undone_at = %s WHERE id = %s", [(undone_at, i) for i in ids])


def claim_calendar_reminders(settings: Settings, reminders: list[dict], fired_at: str, current) -> list[dict]:
    from .calendar import reminders as calendar_reminders
    from .calendar import store as calendar_store
    from datetime import timedelta

    ensure_calendar_schema(settings)
    cutoff = current - timedelta(days=calendar_reminders._LEDGER_RETENTION_DAYS)
    sent: list[dict] = []
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT event_id, event_start, lead_minutes, fired_at FROM assistant_calendar_reminders_fired"))
        stale = [
            (r["event_id"], r["event_start"], r["lead_minutes"])
            for r in rows
            if (fired := calendar_store.parse_dt(str(r["fired_at"]))) is None or fired < cutoff
        ]
        _executemany(
            conn,
            "DELETE FROM assistant_calendar_reminders_fired WHERE event_id = %s AND event_start = %s AND lead_minutes = %s",
            stale,
        )
        for reminder in reminders:
            claimed = 0
            for lead in reminder["covered_leads"]:
                cur = conn.execute(
                    "INSERT INTO assistant_calendar_reminders_fired (event_id, event_start, lead_minutes, fired_at) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING event_id",
                    (reminder["event_id"], reminder["start"], lead, fired_at),
                )
                claimed += 1 if cur.fetchone() else 0
            if claimed:
                sent.append(reminder)
    return sent


def claim_task_reminders(settings: Settings, reminders: list[dict], fired_at: str, current) -> list[dict]:
    from .tasks import reminders as task_reminders
    from .calendar.store import parse_dt
    from datetime import timedelta

    ensure_tasks_schema(settings)
    cutoff = current - timedelta(days=task_reminders._LEDGER_RETENTION_DAYS)
    sent: list[dict] = []
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT task_id, due, lead_minutes, fired_at FROM assistant_task_reminders_fired"))
        stale = [
            (r["task_id"], r["due"], r["lead_minutes"])
            for r in rows
            if (fired := parse_dt(str(r["fired_at"]))) is None or fired < cutoff
        ]
        _executemany(
            conn,
            "DELETE FROM assistant_task_reminders_fired WHERE task_id = %s AND due = %s AND lead_minutes = %s",
            stale,
        )
        for reminder in reminders:
            claimed = 0
            for lead in reminder["covered_leads"]:
                cur = conn.execute(
                    "INSERT INTO assistant_task_reminders_fired (task_id, due, lead_minutes, fired_at) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING task_id",
                    (reminder["task_id"], reminder["due"], lead, fired_at),
                )
                claimed += 1 if cur.fetchone() else 0
            if claimed:
                sent.append(reminder)
    return sent



def ensure_telegram_schema(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_telegram_chats (
              chat_id BIGINT PRIMARY KEY,
              paired_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def paired_telegram_chats(settings: Settings) -> list[int]:
    ensure_telegram_schema(settings)
    with connect(settings) as conn:
        rows = conn.execute("SELECT chat_id FROM assistant_telegram_chats ORDER BY chat_id").fetchall()
    return [int(row[0]) for row in rows]


def pair_telegram_chat(settings: Settings, chat_id: int) -> None:
    ensure_telegram_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_telegram_chats(chat_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (int(chat_id),),
        )
