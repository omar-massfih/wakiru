"""Shared SQLite connection plumbing for the local stores.

Every local store opens a fresh connection per operation with the same
discipline — WAL journalling and a busy timeout, under the memory directory —
and wraps each call in one transaction on a connection that is closed on exit
(``with sqlite3.connect(...)`` alone commits but never closes, leaving cleanup
to CPython refcounting). These two helpers hold that shared shape so each store
keeps only its own schema/DDL.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def open_db(path: str | Path, *, row_factory: bool = True) -> sqlite3.Connection:
    """Open ``path`` with WAL + a busy timeout, creating its parent directory.

    The caller runs its own ``CREATE TABLE IF NOT EXISTS`` / migrations on the
    returned connection. Pass ``row_factory=False`` for stores that read rows
    positionally (see :mod:`assistant.memory.index`).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    if row_factory:
        conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def ensure_columns(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...]
) -> None:
    """Add ``TEXT DEFAULT ''`` columns missing from ``table`` (cheap migration).

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a DB
    created before a column existed would lack it. Reads the name positionally
    (``row[1]``) so it works with or without a row factory.
    """
    have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for column in columns:
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT DEFAULT ''")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One transaction on ``conn``, committed on clean exit and always closed.

    Mirrors the ``with _connect(settings) as conn`` shape every store already
    uses: ``with conn`` commits (or rolls back on exception), and the ``finally``
    closes the connection deterministically rather than on refcount.
    """
    try:
        with conn:
            yield conn
    finally:
        conn.close()
