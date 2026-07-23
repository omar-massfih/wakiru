"""Trips table for the Postgres backend (twin of assistant.trips.store)."""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect

_COLS = "id, destination, name, start, \"end\", timezone, notes, created, updated"


def ensure_trips_schema(settings: Settings) -> None:
    if _schema_done(settings, "trips"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_trips (
              id TEXT PRIMARY KEY,
              destination TEXT NOT NULL,
              name TEXT NOT NULL DEFAULT '',
              start TEXT NOT NULL DEFAULT '',
              "end" TEXT NOT NULL DEFAULT '',
              timezone TEXT NOT NULL DEFAULT '',
              notes TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "trips")


def _trip_from_row(row: dict):
    from ..trips.store import Trip

    return Trip(
        id=str(row["id"]),
        destination=str(row["destination"]),
        name=str(row.get("name") or ""),
        start=str(row.get("start") or ""),
        end=str(row.get("end") or ""),
        timezone=str(row.get("timezone") or ""),
        notes=str(row.get("notes") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
    )


def create_trip(
    settings: Settings,
    destination: str,
    name: str = "",
    start: str = "",
    end: str = "",
    timezone: str = "",
    notes: str = "",
):
    import uuid

    from ..trips import store as trips_store

    ensure_trips_schema(settings)
    now = trips_store._stamp_now(settings)
    trip = trips_store.Trip(
        id=uuid.uuid4().hex[:12],
        destination=destination.strip(),
        name=name.strip() or destination.strip(),
        start=trips_store._normalize_date(start),
        end=trips_store._normalize_date(end),
        timezone=timezone.strip(),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_trips"
            ' (id, destination, name, start, "end", timezone, notes, created, updated)'
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (trip.id, trip.destination, trip.name, trip.start, trip.end,
             trip.timezone, trip.notes, trip.created, trip.updated),
        )
    return trip


def get_trip(settings: Settings, trip_id: str):
    ensure_trips_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(f"SELECT {_COLS} FROM assistant_trips WHERE id = %s", (trip_id,))
        )
    return _trip_from_row(rows[0]) if rows else None


def list_trips(settings: Settings):
    ensure_trips_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_trips"))
    return [_trip_from_row(r) for r in rows]


def update_trip(settings: Settings, trip_id: str, updates: dict):
    from ..trips import store as trips_store

    ensure_trips_schema(settings)
    existing = get_trip(settings, trip_id)
    if existing is None:
        return None
    if not updates:
        return existing
    updates = dict(updates)
    updates["updated"] = trips_store._stamp_now(settings)
    assignments = ", ".join(
        ('"end"' if k == "end" else k) + " = %s" for k in updates
    )
    with connect(settings) as conn:
        conn.execute(
            f"UPDATE assistant_trips SET {assignments} WHERE id = %s",
            (*updates.values(), trip_id),
        )
    return get_trip(settings, trip_id)


def delete_trip(settings: Settings, trip_id: str):
    existing = get_trip(settings, trip_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_trips WHERE id = %s", (trip_id,))
    return existing
