"""SQLite-backed store for named checklists (shopping, errands, packing …).

A single ``list_items`` table in its own SQLite file
(:attr:`Settings.lists_db_path`, under the memory directory), modeled on
:mod:`assistant.reading.store`: an entry has a ``list_name``, an ``item`` text,
and a ``done`` flag. Lists exist implicitly — a list is the set of entries
sharing a name, matched case-insensitively with the first spelling kept as the
display form. Low-stakes writes (a mis-added item is just removed), so there is
no undo ledger.

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


@dataclass
class ListEntry:
    """One item on one named list.

    ``done`` is the checked-off state; ``done_at`` is the tz-aware ISO stamp
    when it was checked (empty while open). ``created`` orders a list the way
    it was written down (oldest first).
    """

    id: str
    list_name: str
    item: str
    done: bool = False
    created: str = ""
    done_at: str = ""


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.lists_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS list_items ("
        " id TEXT PRIMARY KEY, list_name TEXT NOT NULL, item TEXT NOT NULL,"
        " done INTEGER DEFAULT 0, created TEXT DEFAULT '', done_at TEXT DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see tasks.store)."""
    with transaction(_open(settings)) as conn:
        yield conn


def _row_to_entry(row: sqlite3.Row) -> ListEntry:
    return ListEntry(
        id=row["id"],
        list_name=row["list_name"],
        item=row["item"],
        done=bool(row["done"]),
        created=row["created"] or "",
        done_at=row["done_at"] or "",
    )


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def canonical_name(settings: Settings, name: str) -> str:
    """The stored spelling of ``name`` if a list already matches it
    case-insensitively, else ``name`` as given (stripped)."""
    name = name.strip()
    lowered = name.lower()
    for existing, _count in list_names(settings):
        if existing.lower() == lowered:
            return existing
    return name


def add_item(settings: Settings, list_name: str, item: str) -> ListEntry:
    """Add one item to a named list and return it."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.create_list_item(settings, list_name, item)
    entry = ListEntry(
        id=uuid.uuid4().hex[:12],
        list_name=canonical_name(settings, list_name),
        item=item.strip(),
        created=_stamp_now(settings),
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO list_items (id, list_name, item, done, created, done_at)"
            " VALUES (?, ?, ?, 0, ?, '')",
            (entry.id, entry.list_name, entry.item, entry.created),
        )
    return entry


def get_item(settings: Settings, item_id: str) -> ListEntry | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_list_item(settings, item_id)
    with _connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM list_items WHERE id = ?", (item_id,)
        ).fetchone()
    return _row_to_entry(row) if row else None


def _all_entries(settings: Settings) -> list[ListEntry]:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.list_list_items(settings)
    with _connect(settings) as conn:
        rows = conn.execute("SELECT * FROM list_items").fetchall()
    return [_row_to_entry(r) for r in rows]


def list_names(settings: Settings) -> list[tuple[str, int]]:
    """Every list that has any entry, as ``(display name, open count)``,
    sorted by name. Lists whose items are all checked off still appear
    (open count 0) until their entries are removed."""
    counts: dict[str, list] = {}
    for entry in _all_entries(settings):
        key = entry.list_name.lower()
        bucket = counts.setdefault(key, [entry.list_name, 0])
        if not entry.done:
            bucket[1] += 1
    return sorted(
        ((name, open_count) for name, open_count in counts.values()),
        key=lambda pair: pair[0].lower(),
    )


def list_items(
    settings: Settings, list_name: str, include_done: bool = False
) -> list[ListEntry]:
    """A list's open items in the order they were added; ``include_done``
    appends checked-off ones after them (newest check first)."""
    lowered = list_name.strip().lower()
    entries = [e for e in _all_entries(settings) if e.list_name.lower() == lowered]
    open_items = sorted((e for e in entries if not e.done), key=lambda e: e.created)
    if not include_done:
        return open_items
    done = sorted((e for e in entries if e.done), key=lambda e: e.done_at, reverse=True)
    return open_items + done


def set_done(settings: Settings, item_id: str, done: bool = True) -> ListEntry | None:
    """Flip an entry's checked-off flag (idempotent); ``None`` if absent."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.set_list_item_done(settings, item_id, done)
    existing = get_item(settings, item_id)
    if existing is None:
        return None
    done_at = _stamp_now(settings) if done else ""
    with _connect(settings) as conn:
        conn.execute(
            "UPDATE list_items SET done = ?, done_at = ? WHERE id = ?",
            (1 if done else 0, done_at, item_id),
        )
    return get_item(settings, item_id)


def delete_item(settings: Settings, item_id: str) -> ListEntry | None:
    """Delete an entry by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_list_item(settings, item_id)
    existing = get_item(settings, item_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM list_items WHERE id = ?", (item_id,))
    return existing


def find_items(settings: Settings, query: str, list_name: str = "") -> list[ListEntry]:
    """Candidate entries for ``query``: an exact-id match alone, else every
    case-insensitive substring match on the item text — scoped to ``list_name``
    when given. Open entries shadow done ones (done entries are only returned
    when nothing open matches)."""
    query = query.strip()
    if not query:
        return []
    exact = get_item(settings, query)
    if exact is not None:
        return [exact]
    needle = query.lower()
    scope = list_name.strip().lower()
    matches = [
        e
        for e in _all_entries(settings)
        if needle in e.item.lower()
        and (not scope or e.list_name.lower() == scope)
    ]
    if not matches:
        return []
    open_items = [e for e in matches if not e.done]
    return open_items or matches


def find_item(settings: Settings, query: str, list_name: str = "") -> ListEntry | None:
    """Resolve ``query`` to a single entry: by exact id, else the best match."""
    matches = find_items(settings, query, list_name)
    return matches[0] if matches else None
