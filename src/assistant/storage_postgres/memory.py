"""Memory notes + semantic index tables for the Postgres backend."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..config import Settings
from ..memory.embeddings import embedding_signature
from ..memory.store import INDEX_FILENAME, Note
from .core import (
    _executemany,
    _rows,
    _schema_done,
    _schema_mark,
    connect,
    vector_literal,
)


def ensure_memory_schema(settings: Settings) -> None:
    if _schema_done(settings, "memory"):
        return
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
              recall_count INTEGER NOT NULL DEFAULT 0,
              relations JSONB NOT NULL DEFAULT '[]'::jsonb
            )
            """
        )
        # Add the relations column to a table created before it existed.
        conn.execute(
            "ALTER TABLE assistant_memory_notes "
            "ADD COLUMN IF NOT EXISTS relations JSONB NOT NULL DEFAULT '[]'::jsonb"
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
    _schema_mark(settings, "memory")


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
        relations=list(row.get("relations") or []),
    )


def write_note(settings: Settings, note: Note) -> Path:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_memory_notes
              (name, description, body, kind, salience, confidence, tags, source,
               created, updated, last_recalled, recall_count, relations)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb)
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
              recall_count = excluded.recall_count,
              relations = excluded.relations
            """,
            (
                note.name,
                note.description,
                note.body,
                note.kind,
                note.salience,
                note.confidence,
                json.dumps(note.tags),
                note.source,
                note.created,
                note.updated,
                note.last_recalled,
                note.recall_count,
                json.dumps(note.relations),
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
            " source, created, updated, last_recalled, recall_count, relations"
            " FROM assistant_memory_notes ORDER BY name"
        )
        return [_note_from_row(row) for row in _rows(cur)]


def find_note(settings: Settings, name: str) -> Note | None:
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT name, description, body, kind, salience, confidence, tags,"
            " source, created, updated, last_recalled, recall_count, relations"
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
                json.dumps(note.tags),
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


def meta_get(conn, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM assistant_memory_meta WHERE key = %s", (key,)
    ).fetchone()
    return row[0] if row else None


def meta_set(conn, key: str, value: str) -> None:
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
        dim = meta_get(conn, "dim")
        if dim is None:
            meta_set(conn, "dim", str(len(vector)))
            meta_set(conn, "embedding_model", embedding_signature(settings))
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


def memory_index_is_empty(settings: Settings) -> bool:
    """Whether the memory index has no rows — the Postgres twin of
    :func:`assistant.memory.index.is_empty`, letting recall skip the query embed."""
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        row = conn.execute("SELECT 1 FROM assistant_memory_index LIMIT 1").fetchone()
    return row is None


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
    today = datetime.now(UTC).date().isoformat()
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
    from ..memory import index
    from ..memory.embeddings import embed_passages

    notes = list_notes(settings)
    ensure_memory_schema(settings)
    with connect(settings) as conn:
        stored_model = meta_get(conn, "embedding_model")
        model_changed = stored_model not in (None, embedding_signature(settings))
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
    for (note, text_hash, rc, lr), vector in zip(pending, vectors, strict=True):
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
