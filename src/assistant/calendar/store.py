"""SQLite-backed store for the assistant's local calendar.

A single ``events`` table in its own SQLite file (:attr:`Settings.calendar_db_path`,
under the memory directory). Datetimes are stored as timezone-aware ISO-8601
strings so the offset travels with the value; range filtering and ordering parse
them back to ``datetime`` rather than comparing strings (a raw string sort would
misorder events written under different UTC offsets, e.g. across a DST change).

A fresh connection is opened per operation with WAL + a busy timeout, matching the
pattern used by the memory index and the LangGraph checkpointer, so the store is
safe to touch from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings

# Columns a caller may set on create/update (id + timestamps are managed here).
_FIELDS = ("title", "start", "end", "location", "notes", "rrule")


@dataclass
class Event:
    """A single calendar event. ``start``/``end`` are tz-aware ISO-8601 strings.

    ``rrule`` is an optional RFC 5545 recurrence rule (e.g. ``FREQ=WEEKLY;BYDAY=MO``)
    with ``start`` as its DTSTART. Empty for a one-shot event; when set, this row is
    the series *master* and concrete occurrences are expanded on read
    (see :mod:`assistant.calendar.recurrence`).
    """

    id: str
    title: str
    start: str
    end: str = ""
    location: str = ""
    notes: str = ""
    rrule: str = ""
    created: str = ""
    updated: str = ""


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.calendar_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        " id TEXT PRIMARY KEY, title TEXT NOT NULL, start TEXT NOT NULL,"
        " end TEXT DEFAULT '', location TEXT DEFAULT '', notes TEXT DEFAULT '',"
        " rrule TEXT DEFAULT '',"
        " created TEXT DEFAULT '', updated TEXT DEFAULT '')"
    )
    _ensure_columns(conn)
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the table's first creation (cheap migration).

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a DB created
    before ``rrule`` existed would lack the column. Add it in place when missing.
    """
    have = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
    if "rrule" not in have:
        conn.execute("ALTER TABLE events ADD COLUMN rrule TEXT DEFAULT ''")


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        title=row["title"],
        start=row["start"],
        end=row["end"] or "",
        location=row["location"] or "",
        notes=row["notes"] or "",
        rrule=row["rrule"] or "",
        created=row["created"] or "",
        updated=row["updated"] or "",
    )


def parse_dt(value: str) -> datetime | None:
    """Parse a stored ISO-8601 datetime; ``None`` if blank or malformed."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _sort_key(event: Event) -> tuple[int, float, str]:
    """Order by start instant; events with an unparseable start sort last."""
    dt = parse_dt(event.start)
    if dt is None:
        return (1, 0.0, event.start)
    return (0, dt.timestamp(), event.start)


def create_event(
    settings: Settings,
    title: str,
    start: str,
    end: str = "",
    location: str = "",
    notes: str = "",
    rrule: str = "",
) -> Event:
    """Insert a new event and return it (with a generated id and timestamps)."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    event = Event(
        id=uuid.uuid4().hex[:12],
        title=title.strip(),
        start=start.strip(),
        end=end.strip(),
        location=location.strip(),
        notes=notes.strip(),
        rrule=rrule.strip(),
        created=now,
        updated=now,
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO events"
            " (id, title, start, end, location, notes, rrule, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id, event.title, event.start, event.end,
                event.location, event.notes, event.rrule, event.created, event.updated,
            ),
        )
    return event


def get_event(settings: Settings, event_id: str) -> Event | None:
    with _connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
    return _row_to_event(row) if row else None


def list_events(
    settings: Settings,
    start_from: datetime | None = None,
    start_to: datetime | None = None,
) -> list[Event]:
    """Events whose start falls within ``[start_from, start_to]``, soonest first.

    Bounds are optional and inclusive. Events with an unparseable start are
    excluded when either bound is given, and otherwise sorted to the end.
    """
    with _connect(settings) as conn:
        rows = conn.execute("SELECT * FROM events").fetchall()
    events = [_row_to_event(r) for r in rows]

    if start_from is not None or start_to is not None:
        bounded: list[Event] = []
        for event in events:
            dt = parse_dt(event.start)
            if dt is None:
                continue
            if start_from is not None and dt < start_from:
                continue
            if start_to is not None and dt > start_to:
                continue
            bounded.append(event)
        events = bounded

    return sorted(events, key=_sort_key)


def update_event(settings: Settings, event_id: str, **fields: str) -> Event | None:
    """Update the given columns on an event; return it, or ``None`` if absent.

    Only known, non-``None`` fields in :data:`_FIELDS` are applied.
    """
    updates = {
        k: str(v).strip()
        for k, v in fields.items()
        if k in _FIELDS and v is not None
    }
    existing = get_event(settings, event_id)
    if existing is None:
        return None
    if not updates:
        return existing

    updates["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    columns = ", ".join(f"{k} = ?" for k in updates)
    with _connect(settings) as conn:
        conn.execute(
            f"UPDATE events SET {columns} WHERE id = ?",
            (*updates.values(), event_id),
        )
    return get_event(settings, event_id)


def delete_event(settings: Settings, event_id: str) -> Event | None:
    """Delete an event by id; return it if it existed."""
    existing = get_event(settings, event_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    return existing


def find_event(settings: Settings, query: str) -> Event | None:
    """Resolve ``query`` to a single event: by exact id, else by title match.

    The title fallback is a case-insensitive substring match; when several match,
    the soonest upcoming one wins (past events are only considered if nothing
    upcoming matches), so "move the dentist" targets the next dentist appointment.
    """
    query = query.strip()
    if not query:
        return None

    exact = get_event(settings, query)
    if exact is not None:
        return exact

    needle = query.lower()
    matches = [e for e in list_events(settings) if needle in e.title.lower()]
    if not matches:
        return None

    now = datetime.now().astimezone()
    upcoming = [e for e in matches if (dt := parse_dt(e.start)) and dt >= now]
    return (upcoming or matches)[0]
