"""Checklist table for the Postgres backend (twin of assistant.lists.store)."""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect

_COLS = "id, list_name, item, done, created, done_at"


def ensure_lists_schema(settings: Settings) -> None:
    if _schema_done(settings, "lists"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_list_items (
              id TEXT PRIMARY KEY,
              list_name TEXT NOT NULL,
              item TEXT NOT NULL,
              done BOOLEAN NOT NULL DEFAULT FALSE,
              created TEXT NOT NULL DEFAULT '',
              done_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "lists")


def _entry_from_row(row: dict):
    from ..lists.store import ListEntry

    return ListEntry(
        id=str(row["id"]),
        list_name=str(row["list_name"]),
        item=str(row["item"]),
        done=bool(row.get("done")),
        created=str(row.get("created") or ""),
        done_at=str(row.get("done_at") or ""),
    )


def create_list_item(settings: Settings, list_name: str, item: str):
    import uuid

    from ..lists import store as lists_store

    ensure_lists_schema(settings)
    entry = lists_store.ListEntry(
        id=uuid.uuid4().hex[:12],
        list_name=lists_store.canonical_name(settings, list_name),
        item=item.strip(),
        created=lists_store._stamp_now(settings),
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_list_items (id, list_name, item, done, created, done_at)"
            " VALUES (%s, %s, %s, FALSE, %s, '')",
            (entry.id, entry.list_name, entry.item, entry.created),
        )
    return entry


def get_list_item(settings: Settings, item_id: str):
    ensure_lists_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(
                f"SELECT {_COLS} FROM assistant_list_items WHERE id = %s", (item_id,)
            )
        )
    return _entry_from_row(rows[0]) if rows else None


def list_list_items(settings: Settings):
    ensure_lists_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_list_items"))
    return [_entry_from_row(r) for r in rows]


def set_list_item_done(settings: Settings, item_id: str, done: bool = True):
    from ..lists import store as lists_store

    existing = get_list_item(settings, item_id)
    if existing is None:
        return None
    done_at = lists_store._stamp_now(settings) if done else ""
    with connect(settings) as conn:
        conn.execute(
            "UPDATE assistant_list_items SET done = %s, done_at = %s WHERE id = %s",
            (done, done_at, item_id),
        )
    return get_list_item(settings, item_id)


def delete_list_item(settings: Settings, item_id: str):
    existing = get_list_item(settings, item_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_list_items WHERE id = %s", (item_id,))
    return existing
