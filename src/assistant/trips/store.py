"""SQLite-backed store for trips (travel with a start and end date).

A single ``trips`` table in its own SQLite file (:attr:`Settings.trips_db_path`,
under the memory directory), modeled on :mod:`assistant.subscriptions.store`:
a trip has a ``destination``, inclusive ``start``/``end`` dates (``YYYY-MM-DD``),
an optional IANA ``timezone`` for destination local time, and free-text
``notes`` (flight numbers, hotel, who with). Low-stakes writes, so no undo
ledger — a mis-added trip is just removed.

A fresh connection is opened per operation with WAL + a busy timeout, so the
store is safe to touch from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from datetime import date, datetime, time, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import Settings, postgres_backend
from ..sqlite_util import connect, open_db
from ..timeutil import parse_date
from ..timeutil import stamp_now as _stamp_now
from ..timeutil import today as _today

# Text columns a caller may set on update.
_TEXT_FIELDS = ("name", "destination", "start", "end", "timezone", "notes")


@dataclass
class Trip:
    """One trip.

    ``name`` falls back to the destination. ``start``/``end`` are inclusive
    ``YYYY-MM-DD`` dates; a trip is *active* while today is inside them.
    ``timezone`` is an optional IANA name for destination local time.
    """

    id: str
    destination: str
    name: str = ""
    start: str = ""
    end: str = ""
    timezone: str = ""
    notes: str = ""
    created: str = ""
    updated: str = ""


def valid_timezone(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return True  # optional
    try:
        ZoneInfo(value)
    except Exception:
        return False
    return True


def _normalize_date(value: str) -> str:
    d = parse_date(value)
    return d.isoformat() if d is not None else (value or "").strip()


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.trips_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trips ("
        " id TEXT PRIMARY KEY, destination TEXT NOT NULL, name TEXT DEFAULT '',"
        " start TEXT DEFAULT '', end TEXT DEFAULT '', timezone TEXT DEFAULT '',"
        " notes TEXT DEFAULT '', created TEXT DEFAULT '', updated TEXT DEFAULT '')"
    )
    return conn


def _connect(settings: Settings) -> AbstractContextManager[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see sqlite_util.connect)."""
    return connect(_open, settings)


def _row_to_trip(row: sqlite3.Row) -> Trip:
    return Trip(
        id=row["id"],
        destination=row["destination"],
        name=row["name"] or "",
        start=row["start"] or "",
        end=row["end"] or "",
        timezone=row["timezone"] or "",
        notes=row["notes"] or "",
        created=row["created"] or "",
        updated=row["updated"] or "",
    )


def create_trip(
    settings: Settings,
    destination: str,
    name: str = "",
    start: str = "",
    end: str = "",
    timezone: str = "",
    notes: str = "",
) -> Trip:
    """Insert a trip and return it (with a generated id and timestamps)."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.create_trip(
            settings, destination, name, start, end, timezone, notes
        )
    now = _stamp_now(settings)
    trip = Trip(
        id=uuid.uuid4().hex[:12],
        destination=destination.strip(),
        name=name.strip() or destination.strip(),
        start=_normalize_date(start),
        end=_normalize_date(end),
        timezone=timezone.strip(),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO trips"
            " (id, destination, name, start, end, timezone, notes, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trip.id, trip.destination, trip.name, trip.start, trip.end,
             trip.timezone, trip.notes, trip.created, trip.updated),
        )
    return trip


def get_trip(settings: Settings, trip_id: str) -> Trip | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_trip(settings, trip_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM trips WHERE id = ?", (trip_id,)).fetchone()
    return _row_to_trip(row) if row else None


def list_trips(settings: Settings, include_past: bool = False, today: date | None = None) -> list[Trip]:
    """Trips soonest-first; past ones (ended before ``today``) appended only
    with ``include_past`` (most recently ended first)."""
    if storage_postgres := postgres_backend(settings):
        trips = storage_postgres.list_trips(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM trips").fetchall()
        trips = [_row_to_trip(r) for r in rows]
    cutoff = today if today is not None else _today(settings)
    def is_past(trip: Trip) -> bool:
        ended = parse_date(trip.end)
        return ended is not None and ended < cutoff
    current = sorted(
        (t for t in trips if not is_past(t)), key=lambda t: (t.start or "9999", t.created)
    )
    if not include_past:
        return current
    past = sorted((t for t in trips if is_past(t)), key=lambda t: t.end, reverse=True)
    return current + past


def update_trip(settings: Settings, trip_id: str, **fields: object) -> Trip | None:
    """Update a trip's text fields; return it, or ``None`` if absent."""
    updates = {
        k: str(v).strip() for k, v in fields.items() if k in _TEXT_FIELDS and v is not None
    }
    for key in ("start", "end"):
        if key in updates:
            updates[key] = _normalize_date(updates[key])
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.update_trip(settings, trip_id, updates)
    existing = get_trip(settings, trip_id)
    if existing is None:
        return None
    if not updates:
        return existing
    updates["updated"] = _stamp_now(settings)
    columns = ", ".join(f"{k} = ?" for k in updates)
    with _connect(settings) as conn:
        conn.execute(
            f"UPDATE trips SET {columns} WHERE id = ?", (*updates.values(), trip_id)
        )
    return get_trip(settings, trip_id)


def delete_trip(settings: Settings, trip_id: str) -> Trip | None:
    """Delete a trip by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_trip(settings, trip_id)
    existing = get_trip(settings, trip_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM trips WHERE id = ?", (trip_id,))
    return existing


def find_trips(settings: Settings, query: str) -> list[Trip]:
    """Candidate trips for ``query``: an exact-id match alone, else every
    case-insensitive substring match on the name or destination. Current and
    upcoming trips shadow past ones."""
    query = query.strip()
    if not query:
        return []
    exact = get_trip(settings, query)
    if exact is not None:
        return [exact]
    needle = query.lower()
    everything = list_trips(settings, include_past=True)
    matches = [
        t
        for t in everything
        if needle in t.name.lower() or needle in t.destination.lower()
    ]
    if not matches:
        return []
    current = list_trips(settings)
    current_ids = {t.id for t in current}
    live = [t for t in matches if t.id in current_ids]
    return live or matches


def find_trip(settings: Settings, query: str) -> Trip | None:
    """Resolve ``query`` to a single trip: by exact id, else the best match."""
    matches = find_trips(settings, query)
    return matches[0] if matches else None


def active_trip(settings: Settings, today: date | None = None) -> Trip | None:
    """The trip whose inclusive date range contains ``today``, if any."""
    if today is None:
        today = _today(settings)
    for trip in list_trips(settings, today=today):
        started, ends = parse_date(trip.start), parse_date(trip.end)
        if started is not None and ends is not None and started <= today <= ends:
            return trip
    return None


def active_trip_at(settings: Settings, instant: datetime, home_tz: tzinfo) -> Trip | None:
    """Return the trip active at an absolute instant.

    Timezone-bearing trip ranges are destination-local calendar dates.  A trip
    with no timezone (or a corrupt legacy timezone) uses the home-local date,
    which keeps it visible without changing the assistant's clock.
    """
    if instant.tzinfo is None:
        raise ValueError("instant must be timezone-aware")

    # Ask for all rows with an explicit cutoff so this path never calls
    # timeutil.today(), which would recurse through resolve_tz().  Re-sort the
    # complete set with the store's existing soonest-first key: a destination
    # may still be on its final date after that date has ended at home.
    home_date = instant.astimezone(home_tz).date()
    trips = sorted(
        list_trips(settings, include_past=True, today=home_date),
        key=lambda trip: (trip.start or "9999", trip.created),
    )
    for trip in trips:
        started, ends = parse_date(trip.start), parse_date(trip.end)
        if started is None or ends is None:
            continue
        trip_tz = home_tz
        if trip.timezone:
            with suppress(ValueError, ZoneInfoNotFoundError):
                trip_tz = ZoneInfo(trip.timezone)
        local_date = instant.astimezone(trip_tz).date()
        if started <= local_date <= ends:
            return trip
    return None


def next_trip_at(settings: Settings, instant: datetime, home_tz: tzinfo) -> Trip | None:
    """Return the next trip whose local start boundary is after ``instant``.

    Unlike :func:`next_trip`, this comparison is made against an absolute
    instant.  Each destination-local start date therefore remains upcoming
    until midnight in that trip's timezone, even when home time has already
    advanced to the same calendar date.
    """
    if instant.tzinfo is None:
        raise ValueError("instant must be timezone-aware")

    home_date = instant.astimezone(home_tz).date()
    candidates: list[tuple[datetime, str, str, Trip]] = []
    for trip in list_trips(settings, include_past=True, today=home_date):
        started = parse_date(trip.start)
        if started is None:
            continue
        trip_tz = home_tz
        if trip.timezone:
            with suppress(ValueError, ZoneInfoNotFoundError):
                trip_tz = ZoneInfo(trip.timezone)
        boundary = datetime.combine(started, time.min, tzinfo=trip_tz)
        if boundary > instant:
            candidates.append((boundary, trip.start or "9999", trip.created, trip))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[-1]


def next_trip(settings: Settings, today: date | None = None) -> Trip | None:
    """The soonest trip that has not started yet, if any."""
    if today is None:
        today = _today(settings)
    for trip in list_trips(settings, today=today):
        started = parse_date(trip.start)
        if started is not None and started > today:
            return trip
    return None
