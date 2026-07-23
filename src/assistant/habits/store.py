"""SQLite store for the health / habits log.

A single ``habit_log`` table in its own SQLite file
(:attr:`Settings.habits_db_path`). Each row is one logged entry — a ``habit``
name, an optional numeric ``value`` + ``unit`` (kg, hours, km…), an optional
``note``, and the ``logged_on`` date. Unlike tasks/subscriptions this is an
append log, not a mutable record set; the read path (:mod:`.context`) computes
streaks and trends over it.

A fresh connection is opened per operation with WAL + a busy timeout, so the
store is safe from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime

from ..config import Settings, postgres_backend
from ..sqlite_util import open_db, transaction


@dataclass
class HabitEntry:
    """One logged habit/health entry.

    ``value`` is 0 when the entry is a plain check-in ("went for a run") rather
    than a measurement. ``logged_on`` is a ``YYYY-MM-DD`` date; ``created`` is the
    tz-aware ISO stamp, used to order entries within a day.
    """

    id: str
    habit: str
    value: float = 0.0
    unit: str = ""
    note: str = ""
    logged_on: str = ""
    created: str = ""


def parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _coerce_value(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return default


def _today(settings: Settings) -> date:
    from ..calendar.context import now

    return now(settings).date()


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.habits_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS habit_log ("
        " id TEXT PRIMARY KEY, habit TEXT NOT NULL, value REAL DEFAULT 0,"
        " unit TEXT DEFAULT '', note TEXT DEFAULT '',"
        " logged_on TEXT DEFAULT '', created TEXT DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def _row_to_entry(row: sqlite3.Row) -> HabitEntry:
    return HabitEntry(
        id=row["id"],
        habit=row["habit"],
        value=float(row["value"] or 0.0),
        unit=row["unit"] or "",
        note=row["note"] or "",
        logged_on=row["logged_on"] or "",
        created=row["created"] or "",
    )


def log_entry(
    settings: Settings,
    habit: str,
    value: object = 0,
    unit: str = "",
    note: str = "",
    on: str = "",
) -> HabitEntry:
    """Append one habit entry and return it. ``on`` defaults to today."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.log_habit_entry(settings, habit, _coerce_value(value), unit, note, on)
    logged_on = parse_date(on) or _today(settings)
    entry = HabitEntry(
        id=uuid.uuid4().hex[:12],
        habit=habit.strip(),
        value=_coerce_value(value),
        unit=unit.strip(),
        note=note.strip(),
        logged_on=logged_on.isoformat(),
        created=_stamp_now(settings),
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO habit_log (id, habit, value, unit, note, logged_on, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry.id, entry.habit, entry.value, entry.unit, entry.note,
             entry.logged_on, entry.created),
        )
    return entry


def get_entry(settings: Settings, entry_id: str) -> HabitEntry | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_habit_entry(settings, entry_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM habit_log WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_entry(row) if row else None


def list_entries(settings: Settings, habit: str = "") -> list[HabitEntry]:
    """Entries newest-first (by date, then created). ``habit`` filters by name
    (case-insensitive exact match); empty returns all."""
    if storage_postgres := postgres_backend(settings):
        entries = storage_postgres.list_habit_entries(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM habit_log").fetchall()
        entries = [_row_to_entry(r) for r in rows]
    needle = habit.strip().lower()
    if needle:
        entries = [e for e in entries if e.habit.lower() == needle]
    return sorted(entries, key=lambda e: (e.logged_on, e.created), reverse=True)


def delete_entry(settings: Settings, entry_id: str) -> HabitEntry | None:
    """Delete one logged entry by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_habit_entry(settings, entry_id)
    existing = get_entry(settings, entry_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM habit_log WHERE id = ?", (entry_id,))
    return existing


def habit_names(settings: Settings) -> list[str]:
    """Distinct habit names, most-recently-logged first."""
    seen: dict[str, None] = {}
    for entry in list_entries(settings):  # already newest-first
        seen.setdefault(entry.habit, None)
    return list(seen)
