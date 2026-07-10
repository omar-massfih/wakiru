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

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings

# Columns a caller may set on create/update (id + timestamps are managed here).
_FIELDS = ("title", "start", "end", "location", "notes", "rrule", "exdates", "overrides")

# Columns added after the table's first release, migrated in on connect (see
# :func:`_ensure_columns`). All are TEXT DEFAULT ''.
_ADDED_COLUMNS = ("rrule", "exdates", "overrides")


@dataclass
class Event:
    """A single calendar event. ``start``/``end`` are tz-aware ISO-8601 strings.

    ``rrule`` is an optional RFC 5545 recurrence rule (e.g. ``FREQ=WEEKLY;BYDAY=MO``)
    with ``start`` as its DTSTART. Empty for a one-shot event; when set, this row is
    the series *master* and concrete occurrences are expanded on read
    (see :mod:`assistant.calendar.recurrence`).

    ``exdates`` and ``overrides`` carry per-occurrence exceptions on a series master:
    ``exdates`` is a JSON list of occurrence-start ISO strings to skip; ``overrides``
    is a JSON object mapping an occurrence-start ISO string to the changed fields for
    just that occurrence (a moved/edited single instance). Both empty on a plain event.
    """

    id: str
    title: str
    start: str
    end: str = ""
    location: str = ""
    notes: str = ""
    rrule: str = ""
    exdates: str = ""
    overrides: str = ""
    created: str = ""
    updated: str = ""


def _open(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.calendar_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        " id TEXT PRIMARY KEY, title TEXT NOT NULL, start TEXT NOT NULL,"
        " end TEXT DEFAULT '', location TEXT DEFAULT '', notes TEXT DEFAULT '',"
        " rrule TEXT DEFAULT '', exdates TEXT DEFAULT '', overrides TEXT DEFAULT '',"
        " created TEXT DEFAULT '', updated TEXT DEFAULT '')"
    )
    _ensure_columns(conn)
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit.

    ``with sqlite3.connect(...)`` alone commits but never closes — cleanup
    would ride on CPython refcounting. This keeps every call site's
    ``with _connect(settings) as conn`` shape while closing deterministically.
    """
    conn = _open(settings)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the table's first creation (cheap migration).

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a DB created
    before these columns existed would lack them. Add any that are missing in place.
    """
    have = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
    for column in _ADDED_COLUMNS:
        if column not in have:
            conn.execute(f"ALTER TABLE events ADD COLUMN {column} TEXT DEFAULT ''")


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        title=row["title"],
        start=row["start"],
        end=row["end"] or "",
        location=row["location"] or "",
        notes=row["notes"] or "",
        rrule=row["rrule"] or "",
        exdates=row["exdates"] or "",
        overrides=row["overrides"] or "",
        created=row["created"] or "",
        updated=row["updated"] or "",
    )


def parse_dt(value: str) -> datetime | None:
    """Parse a stored ISO-8601 datetime; ``None`` if blank or malformed.

    Always returns an *aware* datetime: a naive value (a legacy row or hand-edit;
    new writes are normalized by :func:`_normalize_stamp`) is interpreted as
    system-local, so one offset-less string can never make an aware/naive
    comparison raise ``TypeError`` across the read paths.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.astimezone() if dt.tzinfo is None else dt


def _normalize_stamp(settings: Settings, value: str) -> str:
    """Attach the assistant's timezone to a naive ISO datetime on its way in.

    The write-path extractor is told to emit offsets, but an LLM slip must not
    poison the store. Blank or unparseable values pass through unchanged (they
    are filtered on read).
    """
    value = value.strip()
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is not None:
        return value
    # Lazy import: context imports this module at top level.
    from .context import resolve_tz

    return dt.replace(tzinfo=resolve_tz(settings)).isoformat()


def _stamp_now(settings: Settings) -> str:
    """Current time in the assistant's timezone, for created/updated stamps —
    matching how every other stamp in the system is resolved."""
    # Lazy import: context imports this module at top level.
    from .context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


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
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.create_event(settings, title, start, end, location, notes, rrule)
    now = _stamp_now(settings)
    event = Event(
        id=uuid.uuid4().hex[:12],
        title=title.strip(),
        start=_normalize_stamp(settings, start),
        end=_normalize_stamp(settings, end),
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
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.get_event(settings, event_id)
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
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        events = storage_postgres.list_events(settings)
    else:
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


def update_event(settings: Settings, event_id: str, **fields: str | None) -> Event | None:
    """Update the given columns on an event; return it, or ``None`` if absent.

    Only known, non-``None`` fields in :data:`_FIELDS` are applied.
    """
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        updates = {k: str(v).strip() for k, v in fields.items() if k in _FIELDS and v is not None}
        return storage_postgres.update_event(settings, event_id, updates)
    updates = {
        k: str(v).strip()
        for k, v in fields.items()
        if k in _FIELDS and v is not None
    }
    for key in ("start", "end"):
        if key in updates:
            updates[key] = _normalize_stamp(settings, updates[key])
    existing = get_event(settings, event_id)
    if existing is None:
        return None
    if not updates:
        return existing

    updates["updated"] = _stamp_now(settings)
    columns = ", ".join(f"{k} = ?" for k in updates)
    with _connect(settings) as conn:
        conn.execute(
            f"UPDATE events SET {columns} WHERE id = ?",
            (*updates.values(), event_id),
        )
    return get_event(settings, event_id)


def restore_event(settings: Settings, event: Event) -> Event:
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.restore_event(settings, event)
    """Re-insert a full event snapshot verbatim, overwriting any current row
    with the same id.

    Unlike :func:`update_event`, this never bumps ``updated`` — ``created``
    and ``updated`` are taken as-is from ``event``. Used to undo a cancel
    (recreate the deleted row) or to put back a full pre-mutation snapshot
    after a reschedule/skip/move.
    """
    with _connect(settings) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO events"
            " (id, title, start, end, location, notes, rrule, exdates, overrides,"
            " created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id, event.title, event.start, event.end, event.location,
                event.notes, event.rrule, event.exdates, event.overrides,
                event.created, event.updated,
            ),
        )
    return event


def delete_event(settings: Settings, event_id: str) -> Event | None:
    """Delete an event by id; return it if it existed."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.delete_event(settings, event_id)
    existing = get_event(settings, event_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    return existing


def find_events(settings: Settings, query: str) -> list[Event]:
    """All candidate events for ``query``: an exact-id match alone, else every
    case-insensitive title-substring match, soonest first.

    Upcoming matches shadow past ones (past events are only returned when
    nothing upcoming matches), so "the dentist" means the next appointment.
    """
    query = query.strip()
    if not query:
        return []

    exact = get_event(settings, query)
    if exact is not None:
        return [exact]

    needle = query.lower()
    matches = [e for e in list_events(settings) if needle in e.title.lower()]
    if not matches:
        return []

    now = datetime.now().astimezone()
    upcoming = [e for e in matches if (dt := parse_dt(e.start)) and dt >= now]
    return upcoming or matches


def find_event(settings: Settings, query: str) -> Event | None:
    """Resolve ``query`` to a single event: by exact id, else the best title match."""
    matches = find_events(settings, query)
    return matches[0] if matches else None


def load_exdates(event: Event) -> list[str]:
    """The occurrence-start ISO strings skipped on a series (empty if none/malformed)."""
    try:
        data = json.loads(event.exdates or "[]")
    except json.JSONDecodeError:
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


def load_overrides(event: Event) -> dict[str, dict]:
    """The per-occurrence overrides on a series: ISO occurrence-start -> changed fields."""
    try:
        data = json.loads(event.overrides or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def add_exdate(settings: Settings, event_id: str, occurrence: str) -> Event | None:
    """Mark a single occurrence of a series as skipped; return the updated master.

    ``None`` if the event is absent or not a series. Idempotent. Any override
    previously set for this occurrence is dropped, since the occurrence is now gone.
    """
    event = get_event(settings, event_id)
    if event is None or not event.rrule:
        return None
    exdates = load_exdates(event)
    if occurrence not in exdates:
        exdates.append(occurrence)
    overrides = load_overrides(event)
    overrides.pop(occurrence, None)
    return update_event(
        settings, event_id,
        exdates=json.dumps(exdates), overrides=json.dumps(overrides),
    )


def set_override(
    settings: Settings, event_id: str, occurrence: str, fields: dict[str, str]
) -> Event | None:
    """Override the given fields for a single occurrence of a series (a moved instance).

    ``fields`` may include ``start``/``end``/``title``/``location``/``notes``; blanks
    are ignored. Returns the updated master, or ``None`` if absent/not a series.
    """
    event = get_event(settings, event_id)
    if event is None or not event.rrule:
        return None
    kept = {k: str(v).strip() for k, v in fields.items() if k in _FIELDS and v}
    if not kept:
        return event
    overrides = load_overrides(event)
    overrides[occurrence] = {**overrides.get(occurrence, {}), **kept}
    exdates = [d for d in load_exdates(event) if d != occurrence]  # un-skip if moved back
    return update_event(
        settings, event_id,
        overrides=json.dumps(overrides), exdates=json.dumps(exdates),
    )
