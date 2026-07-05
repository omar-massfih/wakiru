"""Vector index over the memory notes, stored in SQLite via sqlite-vec.

Python owns this index so it can never drift from the markdown files: every write
to :mod:`.store` is mirrored here. A fresh connection is opened per operation so
the index is safe to touch from FastAPI request handlers and background tasks
alike.

The ``vec_notes`` virtual table is created lazily on the first ``upsert`` (its
vector dimension is taken from that first vector), so an empty store needs no
schema and ``search`` simply returns nothing.
"""

from __future__ import annotations

import sqlite3
import struct

import sqlite_vec

from ..config import Settings

VEC_TABLE = "vec_notes"


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.memory_db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " name TEXT UNIQUE, path TEXT, description TEXT)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def _vec_dim(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
    return int(row[0]) if row else None


def _ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    if _vec_dim(conn) is not None:
        return
    conn.execute(
        f"CREATE VIRTUAL TABLE {VEC_TABLE} USING "
        f"vec0(embedding float[{dim}] distance_metric=cosine)"
    )
    conn.execute("INSERT INTO meta(key, value) VALUES ('dim', ?)", (str(dim),))


def upsert(
    settings: Settings, name: str, path: str, description: str, vector: list[float]
) -> None:
    """Insert or replace the index entry for ``name``."""
    conn = _connect(settings)
    try:
        _ensure_vec_table(conn, len(vector))
        row = conn.execute("SELECT id FROM notes WHERE name = ?", (name,)).fetchone()
        if row is not None:
            conn.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid = ?", (row[0],))
            conn.execute("DELETE FROM notes WHERE id = ?", (row[0],))
        cur = conn.execute(
            "INSERT INTO notes(name, path, description) VALUES (?, ?, ?)",
            (name, path, description),
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


def search(
    settings: Settings, query_vector: list[float], k: int
) -> list[tuple[str, str, str, float]]:
    """Top-k nearest notes as ``(name, path, description, similarity)``.

    ``similarity`` is cosine similarity in ``[-1, 1]`` (``1 - distance``).
    """
    conn = _connect(settings)
    try:
        if _vec_dim(conn) is None:
            return []
        rows = conn.execute(
            f"SELECT n.name, n.path, n.description, v.distance "
            f"FROM {VEC_TABLE} v JOIN notes n ON n.id = v.rowid "
            f"WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (_serialize(query_vector), k),
        ).fetchall()
        return [(name, path, desc, 1.0 - dist) for name, path, desc, dist in rows]
    finally:
        conn.close()
