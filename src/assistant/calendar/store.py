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
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings, postgres_backend
from ..sqlite_util import connect, ensure_columns, open_db
from ..timeutil import normalize_stamp as _normalize_stamp
from ..timeutil import stamp_now as _stamp_now

# Columns a caller may set on create/update (id + timestamps are managed here).
_FIELDS = ("title", "start", "end", "location", "notes", "rrule", "exdates", "overrides")

# Columns added after the table's first release, migrated in on connect (see
# :func:`assistant.sqlite_util.ensure_columns`). All are TEXT DEFAULT ''.
_ADDED_COLUMNS = (
    "rrule", "exdates", "overrides", "organizer", "attendees",
    "caldav_href", "caldav_etag",
)


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

    ``organizer`` and ``attendees`` are stable provider-neutral JSON retained from
    imported calendars. They carry participant email identity and common scheduling
    attributes; malformed legacy values are treated as empty by the load helpers.

    ``caldav_href``/``caldav_etag`` map a CalDAV-backed row to its remote resource:
    ``caldav_href`` is the server path, ``caldav_etag`` the last-seen validator used as
    the ``If-Match`` precondition on update/delete. Both empty on a purely-local or
    ICS-mirrored event (see :mod:`assistant.calendar.caldav`).
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
    caldav_href: str = ""
    caldav_etag: str = ""
    created: str = ""
    updated: str = ""
    organizer: str = ""
    attendees: str = ""


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.calendar_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        " id TEXT PRIMARY KEY, title TEXT NOT NULL, start TEXT NOT NULL,"
        " end TEXT DEFAULT '', location TEXT DEFAULT '', notes TEXT DEFAULT '',"
        " rrule TEXT DEFAULT '', exdates TEXT DEFAULT '', overrides TEXT DEFAULT '',"
        " organizer TEXT DEFAULT '', attendees TEXT DEFAULT '',"
        " caldav_href TEXT DEFAULT '', caldav_etag TEXT DEFAULT '',"
        " created TEXT DEFAULT '', updated TEXT DEFAULT '')"
    )
    ensure_columns(conn, "events", _ADDED_COLUMNS)
    return conn


def _connect(settings: Settings) -> AbstractContextManager[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see sqlite_util.connect)."""
    return connect(_open, settings)


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
        organizer=row["organizer"] or "",
        attendees=row["attendees"] or "",
        caldav_href=row["caldav_href"] or "",
        caldav_etag=row["caldav_etag"] or "",
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
    if storage_postgres := postgres_backend(settings):
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
    if storage_postgres := postgres_backend(settings):
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
    if storage_postgres := postgres_backend(settings):
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
    if storage_postgres := postgres_backend(settings):
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
    """Re-insert a full event snapshot verbatim, overwriting any current row
    with the same id.

    Unlike :func:`update_event`, this never bumps ``updated`` — ``created``
    and ``updated`` are taken as-is from ``event``. Used to undo a cancel
    (recreate the deleted row) or to put back a full pre-mutation snapshot
    after a reschedule/skip/move.
    """
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.restore_event(settings, event)
    with _connect(settings) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO events"
            " (id, title, start, end, location, notes, rrule, exdates, overrides,"
            " organizer, attendees, caldav_href, caldav_etag, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id, event.title, event.start, event.end, event.location,
                event.notes, event.rrule, event.exdates, event.overrides,
                event.organizer, event.attendees,
                event.caldav_href, event.caldav_etag, event.created, event.updated,
            ),
        )
    return event


def delete_event(settings: Settings, event_id: str) -> Event | None:
    """Delete an event by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_event(settings, event_id)
    existing = get_event(settings, event_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    return existing


def set_caldav_meta(settings: Settings, event_id: str, href: str, etag: str) -> None:
    """Record the remote resource an event now maps to (CalDAV push result).

    A no-op if the event is absent. ``updated`` is left untouched — this is
    bookkeeping about where the row lives remotely, not a content change.
    """
    if storage_postgres := postgres_backend(settings):
        storage_postgres.set_caldav_meta(settings, event_id, href, etag)
        return
    with _connect(settings) as conn:
        conn.execute(
            "UPDATE events SET caldav_href = ?, caldav_etag = ? WHERE id = ?",
            (href, etag, event_id),
        )


def find_by_href(settings: Settings, href: str) -> Event | None:
    """The event mapped to a CalDAV resource path, or ``None`` (empty href never matches)."""
    if not href:
        return None
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.find_by_href(settings, href)
    with _connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE caldav_href = ?", (href,)
        ).fetchone()
    return _row_to_event(row) if row else None


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


def normalize_email(value: object) -> str:
    """A provider calendar address as a lowercase bare email address."""
    email = str(value or "").strip()
    if email.lower().startswith("mailto:"):
        email = email[7:].strip()
    return email.lower()


def _canonical_participant(value: object) -> dict:
    """Normalize one provider-neutral participant, or return an empty value."""
    if not isinstance(value, dict):
        return {}
    email = normalize_email(value.get("email"))
    if not email:
        return {}
    participant: dict = {"email": email}
    for key in ("name", "status", "role"):
        item = str(value.get(key) or "").strip()
        if item:
            participant[key] = item.lower() if key in ("status", "role") else item
    for key in ("rsvp", "self"):
        flag = value.get(key)
        if isinstance(flag, bool):
            participant[key] = flag
    return participant


def dump_organizer(value: object) -> str:
    """Stable JSON for one organizer; empty when it has no usable email."""
    participant = _canonical_participant(value)
    return json.dumps(participant, sort_keys=True, separators=(",", ":")) if participant else ""


def dump_attendees(value: object) -> str:
    """Stable JSON for an unordered attendee collection.

    Providers may return the same attendees in a different order.  Sort their
    canonical representations so mirrored-field comparisons do not treat that
    as a change; sorting a list retains any intentional duplicate entries.
    """
    if not isinstance(value, list):
        return ""
    attendees = [participant for item in value if (participant := _canonical_participant(item))]
    attendees.sort(key=lambda participant: json.dumps(participant, sort_keys=True, separators=(",", ":")))
    return json.dumps(attendees, sort_keys=True, separators=(",", ":")) if attendees else ""


def load_organizer(event: Event) -> dict:
    try:
        data = json.loads(event.organizer or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return _canonical_participant(data)


def load_attendees(event: Event) -> list[dict]:
    try:
        data = json.loads(event.attendees or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [participant for item in data if (participant := _canonical_participant(item))]


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
