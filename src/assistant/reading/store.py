"""SQLite-backed store for the read-it-later list.

A single ``reading`` table in its own SQLite file (:attr:`Settings.reading_db_path`,
under the memory directory), modeled on :mod:`assistant.tasks.store` but simpler:
a saved link has a ``url``, an optional ``title`` and ``note``, and a ``read``
flag. Low-stakes writes (a mis-saved link is just removed), so — unlike tasks and
people — there is no undo ledger.

A fresh connection is opened per operation with WAL + a busy timeout, so the
store is safe to touch from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings, postgres_backend
from ..sqlite_util import open_db, transaction

# Text columns a caller may set on update (url is identity; read has its own path).
_TEXT_FIELDS = ("title", "note")


@dataclass
class ReadingItem:
    """One saved link.

    ``title`` falls back to the URL when unknown. ``read`` is the done state;
    ``read_at`` is the tz-aware ISO stamp when it was marked read (empty while
    unread). ``created`` orders the list (newest first).
    """

    id: str
    url: str
    title: str = ""
    note: str = ""
    read: bool = False
    created: str = ""
    read_at: str = ""


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.reading_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reading ("
        " id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT DEFAULT '',"
        " note TEXT DEFAULT '', read INTEGER DEFAULT 0,"
        " created TEXT DEFAULT '', read_at TEXT DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see tasks.store)."""
    with transaction(_open(settings)) as conn:
        yield conn


def _row_to_item(row: sqlite3.Row) -> ReadingItem:
    return ReadingItem(
        id=row["id"],
        url=row["url"],
        title=row["title"] or "",
        note=row["note"] or "",
        read=bool(row["read"]),
        created=row["created"] or "",
        read_at=row["read_at"] or "",
    )


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def create_item(
    settings: Settings, url: str, title: str = "", note: str = ""
) -> ReadingItem:
    """Save a link and return it (with a generated id and timestamp)."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.create_reading_item(settings, url, title, note)
    item = ReadingItem(
        id=uuid.uuid4().hex[:12],
        url=url.strip(),
        title=title.strip() or url.strip(),
        note=note.strip(),
        created=_stamp_now(settings),
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO reading (id, url, title, note, read, created, read_at)"
            " VALUES (?, ?, ?, ?, 0, ?, '')",
            (item.id, item.url, item.title, item.note, item.created),
        )
    return item


def get_item(settings: Settings, item_id: str) -> ReadingItem | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_reading_item(settings, item_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM reading WHERE id = ?", (item_id,)).fetchone()
    return _row_to_item(row) if row else None


def list_items(settings: Settings, include_read: bool = False) -> list[ReadingItem]:
    """Unread items newest-first; ``include_read`` appends read ones after them."""
    if storage_postgres := postgres_backend(settings):
        items = storage_postgres.list_reading_items(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM reading").fetchall()
        items = [_row_to_item(r) for r in rows]
    unread = sorted((i for i in items if not i.read), key=lambda i: i.created, reverse=True)
    if not include_read:
        return unread
    read = sorted((i for i in items if i.read), key=lambda i: i.read_at, reverse=True)
    return unread + read


def update_item(settings: Settings, item_id: str, **fields: str | None) -> ReadingItem | None:
    """Update a saved link's title/note; return it, or ``None`` if absent."""
    if storage_postgres := postgres_backend(settings):
        updates = {k: str(v).strip() for k, v in fields.items() if k in _TEXT_FIELDS and v is not None}
        return storage_postgres.update_reading_item(settings, item_id, updates)
    updates = {k: str(v).strip() for k, v in fields.items() if k in _TEXT_FIELDS and v is not None}
    existing = get_item(settings, item_id)
    if existing is None:
        return None
    if not updates:
        return existing
    columns = ", ".join(f"{k} = ?" for k in updates)
    with _connect(settings) as conn:
        conn.execute(
            f"UPDATE reading SET {columns} WHERE id = ?", (*updates.values(), item_id)
        )
    return get_item(settings, item_id)


def mark_read(settings: Settings, item_id: str, read: bool = True) -> ReadingItem | None:
    """Flip an item's read flag (idempotent); return it, or ``None`` if absent."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.mark_reading_read(settings, item_id, read)
    existing = get_item(settings, item_id)
    if existing is None:
        return None
    read_at = _stamp_now(settings) if read else ""
    with _connect(settings) as conn:
        conn.execute(
            "UPDATE reading SET read = ?, read_at = ? WHERE id = ?",
            (1 if read else 0, read_at, item_id),
        )
    return get_item(settings, item_id)


def delete_item(settings: Settings, item_id: str) -> ReadingItem | None:
    """Delete a saved link by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_reading_item(settings, item_id)
    existing = get_item(settings, item_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM reading WHERE id = ?", (item_id,))
    return existing


def find_items(settings: Settings, query: str) -> list[ReadingItem]:
    """Candidate items for ``query``: an exact-id match alone, else every
    case-insensitive substring match on the title or URL. Unread shadow read
    (read items are only returned when nothing unread matches)."""
    query = query.strip()
    if not query:
        return []
    exact = get_item(settings, query)
    if exact is not None:
        return [exact]
    needle = query.lower()
    matches = [
        i
        for i in list_items(settings, include_read=True)
        if needle in i.title.lower() or needle in i.url.lower()
    ]
    if not matches:
        return []
    unread = [i for i in matches if not i.read]
    return unread or matches


def find_item(settings: Settings, query: str) -> ReadingItem | None:
    """Resolve ``query`` to a single item: by exact id, else the best match."""
    matches = find_items(settings, query)
    return matches[0] if matches else None
