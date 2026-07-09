"""Vector index over the memory notes, stored in SQLite via sqlite-vec.

Python owns this index so it can never drift from the markdown files: every write
to :mod:`.store` is mirrored here, and :func:`reindex` rebuilds the whole thing
from disk (the files are the source of truth). A fresh connection is opened per
operation so the index is safe to touch from FastAPI request handlers and
background tasks alike; WAL mode keeps readers and the background writer from
blocking each other.

The ``vec_notes`` virtual table is created lazily on the first ``upsert`` (its
vector dimension is taken from that first vector), so an empty store needs no
schema and ``search`` simply returns nothing. The ``meta`` table records the
embedding model + dimension so :func:`reindex` can detect a model swap and
rebuild from scratch.

Alongside each vector the ``notes`` table keeps cheap re-ranking columns
(``kind``, ``salience``, ``updated``, and the reinforcement counters
``recall_count`` / ``last_recalled``) so recall can blend signals without opening
every file. The counters are authoritative here and mirrored back to the markdown
frontmatter on consolidation.
"""

from __future__ import annotations

import hashlib
import struct
import sqlite3
from datetime import date

import sqlite_vec

from ..config import Settings
from .locks import locked

VEC_TABLE = "vec_notes"


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def content_hash(text: str) -> str:
    """Fingerprint of the embedded text, so reindex can skip unchanged notes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.memory_db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " name TEXT UNIQUE, path TEXT, description TEXT,"
        " kind TEXT DEFAULT 'semantic', salience REAL DEFAULT 0.5,"
        " updated TEXT DEFAULT '', last_recalled TEXT DEFAULT '',"
        " recall_count INTEGER DEFAULT 0, hash TEXT DEFAULT '')"
    )
    # Migrate pre-hash databases in place (a blank hash just means "re-embed").
    columns = {row[1] for row in conn.execute("PRAGMA table_info(notes)")}
    if "hash" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN hash TEXT DEFAULT ''")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _vec_dim(conn: sqlite3.Connection) -> int | None:
    value = _meta_get(conn, "dim")
    return int(value) if value else None


def _ensure_vec_table(conn: sqlite3.Connection, dim: int, settings: Settings) -> None:
    if _vec_dim(conn) is not None:
        return
    conn.execute(
        f"CREATE VIRTUAL TABLE {VEC_TABLE} USING "
        f"vec0(embedding float[{dim}] distance_metric=cosine)"
    )
    _meta_set(conn, "dim", str(dim))
    _meta_set(conn, "embedding_model", settings.embedding_model)


def upsert(
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
    """Insert or replace the index entry for ``name`` (preserving its rowid data)."""
    conn = _connect(settings)
    try:
        _ensure_vec_table(conn, len(vector), settings)
        row = conn.execute("SELECT id FROM notes WHERE name = ?", (name,)).fetchone()
        if row is not None:
            conn.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid = ?", (row[0],))
            conn.execute("DELETE FROM notes WHERE id = ?", (row[0],))
        cur = conn.execute(
            "INSERT INTO notes(name, path, description, kind, salience, updated,"
            " last_recalled, recall_count, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, path, description, kind, salience, updated, last_recalled,
             recall_count, text_hash),
        )
        conn.execute(
            f"INSERT INTO {VEC_TABLE}(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, _serialize(vector)),
        )
        conn.commit()
    finally:
        conn.close()


def remove(settings: Settings, name: str) -> bool:
    """Drop ``name`` from the index. Returns whether it existed."""
    conn = _connect(settings)
    try:
        row = conn.execute("SELECT id FROM notes WHERE name = ?", (name,)).fetchone()
        if row is None:
            return False
        if _vec_dim(conn) is not None:
            conn.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid = ?", (row[0],))
        conn.execute("DELETE FROM notes WHERE id = ?", (row[0],))
        conn.commit()
        return True
    finally:
        conn.close()


def search_ranked(
    settings: Settings, query_vector: list[float], k: int
) -> list[tuple[str, str, str, str, float, int, str, float]]:
    """Top-k as ``(name, path, description, kind, salience, recall_count,
    last_recalled, similarity)`` — everything recall needs to re-rank cheaply.
    """
    conn = _connect(settings)
    try:
        if _vec_dim(conn) is None:
            return []
        rows = conn.execute(
            f"SELECT n.name, n.path, n.description, n.kind, n.salience,"
            f" n.recall_count, n.last_recalled, v.distance "
            f"FROM {VEC_TABLE} v JOIN notes n ON n.id = v.rowid "
            f"WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (_serialize(query_vector), k),
        ).fetchall()
        return [
            (name, path, desc, kind, sal, rc, lr, 1.0 - dist)
            for name, path, desc, kind, sal, rc, lr, dist in rows
        ]
    finally:
        conn.close()


def list_entries(
    settings: Settings,
) -> list[tuple[str, str, str, float, int, str, str]]:
    """All index rows as ``(name, description, kind, salience, recall_count,
    last_recalled, updated)`` — enough to rank the injected index view and the
    consolidation eviction pass without opening any files. ``salience`` is the
    *effective* value (consolidation may have decayed it below the file's copy).
    """
    conn = _connect(settings)
    try:
        return conn.execute(
            "SELECT name, description, kind, salience, recall_count,"
            " last_recalled, updated FROM notes"
        ).fetchall()
    finally:
        conn.close()


def bump_turn_counter(settings: Settings) -> int:
    """Increment and return the persistent chat-turn counter.

    Drives the periodic-consolidation cadence; lives in the ``meta`` table so it
    survives server restarts (an in-process counter would reset and could starve
    consolidation under frequent restarts).
    """
    conn = _connect(settings)
    try:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('turn_count', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'turn_count'"
        ).fetchone()
        conn.commit()
        return int(row[0])
    finally:
        conn.close()


def bump_recall(settings: Settings, names: list[str]) -> None:
    """Reinforce: increment ``recall_count`` and stamp ``last_recalled`` = today.

    Cheap, index-only — the authoritative counters. Consolidation later mirrors
    them into the markdown frontmatter. No-op for unknown names.
    """
    if not names:
        return
    today = date.today().isoformat()
    conn = _connect(settings)
    try:
        conn.executemany(
            "UPDATE notes SET recall_count = recall_count + 1, last_recalled = ? "
            "WHERE name = ?",
            [(today, name) for name in names],
        )
        conn.commit()
    finally:
        conn.close()


def set_salience(settings: Settings, name: str, salience: float) -> None:
    """Update the cached salience used for re-ranking (index-only)."""
    conn = _connect(settings)
    try:
        conn.execute(
            "UPDATE notes SET salience = ? WHERE name = ?", (float(salience), name)
        )
        conn.commit()
    finally:
        conn.close()


def get_stats(settings: Settings, name: str) -> tuple[int, str] | None:
    """Return ``(recall_count, last_recalled)`` for ``name``, or ``None``."""
    conn = _connect(settings)
    try:
        row = conn.execute(
            "SELECT recall_count, last_recalled FROM notes WHERE name = ?", (name,)
        ).fetchone()
        return (int(row[0]), str(row[1] or "")) if row else None
    finally:
        conn.close()


@locked
def reindex(settings: Settings) -> int:
    """Rebuild the vector index from the markdown files (the source of truth).

    Self-heals drift from hand-edits and migrates on an embedding-model change:

    * If the recorded model/dim differs from ``settings.embedding_model`` the vec
      table is dropped and rebuilt from scratch.
    * A note whose ``index_text`` hash matches its index row keeps its vector and
      only gets a metadata refresh — so a routine restart embeds nothing. Changed
      or new files are (re-)embedded; index rows whose files vanished are removed.
    * Reinforcement counters (``recall_count`` / ``last_recalled``) are preserved
      by name across the rebuild, falling back to the file's own frontmatter when
      the index had no prior value.

    Returns the number of notes indexed.
    """
    from . import store
    from .embeddings import embed_passages

    notes = store.list_notes(settings)

    conn = _connect(settings)
    try:
        # Snapshot existing counters so a full rebuild doesn't lose reinforcement.
        prior: dict[str, tuple[int, str]] = {
            name: (int(rc), str(lr or ""))
            for name, rc, lr in conn.execute(
                "SELECT name, recall_count, last_recalled FROM notes"
            ).fetchall()
        }
        model_changed = _meta_get(conn, "embedding_model") not in (
            None,
            settings.embedding_model,
        )
        if model_changed and _vec_dim(conn) is not None:
            conn.execute(f"DROP TABLE IF EXISTS {VEC_TABLE}")
            conn.execute("DELETE FROM notes")
            conn.execute("DELETE FROM meta WHERE key IN ('dim', 'embedding_model')")
            conn.commit()
    finally:
        conn.close()

    # Split into unchanged (hash matches a live vector row -> metadata refresh
    # only) and changed/new (re-embed). A blank stored hash always re-embeds.
    pending: list[tuple] = []  # (note, hash, recall_count, last_recalled)
    live_names: set[str] = set()
    conn = _connect(settings)
    try:
        vec_ready = _vec_dim(conn) is not None
        for note in notes:
            live_names.add(note.name)
            text_hash = content_hash(note.index_text)
            rc, lr = prior.get(note.name, (note.recall_count, note.last_recalled))
            if not model_changed and vec_ready:
                row = conn.execute(
                    "SELECT id, hash FROM notes WHERE name = ?", (note.name,)
                ).fetchone()
                vec_row = (
                    conn.execute(
                        f"SELECT rowid FROM {VEC_TABLE} WHERE rowid = ?", (row[0],)
                    ).fetchone()
                    if row is not None
                    else None
                )
                if row is not None and vec_row is not None and row[1] == text_hash:
                    # Deliberately NOT refreshing salience: the index carries the
                    # *effective* (possibly decayed) value, which consolidation
                    # maintains index-only — taking the file's copy here would
                    # undo durable decay on every restart. A changed note goes
                    # through upsert below and picks up the file's salience.
                    conn.execute(
                        "UPDATE notes SET path = ?, description = ?, kind = ?,"
                        " updated = ? WHERE id = ?",
                        (str(store.note_path(settings, note)), note.description,
                         note.kind, note.updated, row[0]),
                    )
                    continue
            pending.append((note, text_hash, rc, lr))
        conn.commit()
    finally:
        conn.close()

    # Re-embed only what changed, carrying counters forward.
    vectors = (
        embed_passages([n.index_text for n, _h, _rc, _lr in pending], settings)
        if pending
        else []
    )
    if len(vectors) != len(pending):
        # zip would silently drop the tail — those notes would never be indexed.
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(pending)} notes"
        )
    for (note, text_hash, rc, lr), vector in zip(pending, vectors):
        upsert(
            settings,
            note.name,
            str(store.note_path(settings, note)),
            note.description,
            vector,
            kind=note.kind,
            salience=note.salience,
            updated=note.updated,
            last_recalled=lr,
            recall_count=rc,
            text_hash=text_hash,
        )

    # Drop index rows whose files disappeared.
    conn = _connect(settings)
    try:
        stale = [
            row[0]
            for row in conn.execute("SELECT name FROM notes").fetchall()
            if row[0] not in live_names
        ]
    finally:
        conn.close()
    for name in stale:
        remove(settings, name)

    return len(live_names)
