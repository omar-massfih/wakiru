"""Reading-list table for the Postgres backend (twin of assistant.reading.store)."""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect

_COLS = "id, url, title, note, read, created, read_at"


def ensure_reading_schema(settings: Settings) -> None:
    if _schema_done(settings, "reading"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_reading (
              id TEXT PRIMARY KEY,
              url TEXT NOT NULL,
              title TEXT NOT NULL DEFAULT '',
              note TEXT NOT NULL DEFAULT '',
              read BOOLEAN NOT NULL DEFAULT FALSE,
              created TEXT NOT NULL DEFAULT '',
              read_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "reading")


def _item_from_row(row: dict):
    from ..reading.store import ReadingItem

    return ReadingItem(
        id=str(row["id"]),
        url=str(row["url"]),
        title=str(row.get("title") or ""),
        note=str(row.get("note") or ""),
        read=bool(row.get("read")),
        created=str(row.get("created") or ""),
        read_at=str(row.get("read_at") or ""),
    )


def create_reading_item(settings: Settings, url: str, title: str = "", note: str = ""):
    import uuid

    from ..reading import store as reading_store

    ensure_reading_schema(settings)
    item = reading_store.ReadingItem(
        id=uuid.uuid4().hex[:12],
        url=url.strip(),
        title=title.strip() or url.strip(),
        note=note.strip(),
        created=reading_store._stamp_now(settings),
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_reading (id, url, title, note, read, created, read_at)"
            " VALUES (%s, %s, %s, %s, FALSE, %s, '')",
            (item.id, item.url, item.title, item.note, item.created),
        )
    return item


def get_reading_item(settings: Settings, item_id: str):
    ensure_reading_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(f"SELECT {_COLS} FROM assistant_reading WHERE id = %s", (item_id,))
        )
    return _item_from_row(rows[0]) if rows else None


def list_reading_items(settings: Settings):
    ensure_reading_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_reading"))
    return [_item_from_row(r) for r in rows]


def update_reading_item(settings: Settings, item_id: str, fields: dict):
    ensure_reading_schema(settings)
    existing = get_reading_item(settings, item_id)
    if existing is None:
        return None
    updates = {k: str(v).strip() for k, v in fields.items() if v is not None}
    if not updates:
        return existing
    assignments = ", ".join(f"{k} = %s" for k in updates)
    with connect(settings) as conn:
        conn.execute(
            f"UPDATE assistant_reading SET {assignments} WHERE id = %s",
            (*updates.values(), item_id),
        )
    return get_reading_item(settings, item_id)


def mark_reading_read(settings: Settings, item_id: str, read: bool = True):
    from ..reading import store as reading_store

    existing = get_reading_item(settings, item_id)
    if existing is None:
        return None
    read_at = reading_store._stamp_now(settings) if read else ""
    with connect(settings) as conn:
        conn.execute(
            "UPDATE assistant_reading SET read = %s, read_at = %s WHERE id = %s",
            (read, read_at, item_id),
        )
    return get_reading_item(settings, item_id)


def delete_reading_item(settings: Settings, item_id: str):
    existing = get_reading_item(settings, item_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_reading WHERE id = %s", (item_id,))
    return existing
