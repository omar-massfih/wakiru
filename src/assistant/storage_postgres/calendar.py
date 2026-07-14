"""Calendar event tables for the Postgres backend."""

from __future__ import annotations

from ..config import Settings
from .core import (
    _rows,
    _schema_done,
    _schema_mark,
    connect,
)


def ensure_calendar_schema(settings: Settings) -> None:
    if _schema_done(settings, "calendar"):
        return
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
    _schema_mark(settings, "calendar")


def _event_from_row(row: dict):
    from ..calendar.store import Event

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
    import uuid

    from ..calendar import store as calendar_store

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
    from ..calendar import store as calendar_store

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
