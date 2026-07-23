"""SQLite store for the work log (time tracking).

A single ``work_log`` table in its own SQLite file
(:attr:`Settings.worklog_db_path`). Each row is one stretch of work on a
*project* — either a live timer (``started`` set when the clock starts,
``ended`` + ``minutes`` filled when it stops) or an after-the-fact log
("2 hours on client X yesterday": ``minutes`` given directly, no timer
stamps). Like the habits and expense logs this is an append log the read path
(:mod:`.context`) totals per project; corrections happen by removing an entry,
not editing it.

At most one timer runs at a time: starting a new one stops the running one
first (people switch tasks without closing the old one), and both entries come
back so the tool can report the switch. A fresh connection is opened per
operation with WAL + a busy timeout, so the store is safe from FastAPI request
handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime

from ..calendar.store import parse_dt
from ..config import Settings, postgres_backend
from ..sqlite_util import open_db, transaction


@dataclass
class WorkEntry:
    """One stretch of work.

    ``minutes`` is the duration (0 while a timer is still running).
    ``worked_on`` is the ``YYYY-MM-DD`` date the time counts toward (the day
    the timer started, or the day given on a direct log). ``started``/``ended``
    are tz-aware ISO stamps on timer entries and empty on direct logs; a row
    with ``started`` set and ``ended`` empty is the running timer.
    """

    id: str
    project: str
    minutes: int = 0
    note: str = ""
    worked_on: str = ""
    started: str = ""
    ended: str = ""
    created: str = ""


def parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def coerce_minutes(value: object, default: int = 0) -> int:
    """A minutes arg ("90", "90.5", 90) as a non-negative int; default when junk."""
    try:
        minutes = round(float(str(value).strip().replace(",", ".")))
    except (TypeError, ValueError):
        return default
    return minutes if minutes > 0 else default


def _today(settings: Settings) -> date:
    from ..calendar.context import now

    return now(settings).date()


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.worklog_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS work_log ("
        " id TEXT PRIMARY KEY, project TEXT NOT NULL, minutes INTEGER DEFAULT 0,"
        " note TEXT DEFAULT '', worked_on TEXT DEFAULT '',"
        " started TEXT DEFAULT '', ended TEXT DEFAULT '', created TEXT DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def _row_to_entry(row: sqlite3.Row) -> WorkEntry:
    return WorkEntry(
        id=row["id"],
        project=row["project"],
        minutes=int(row["minutes"] or 0),
        note=row["note"] or "",
        worked_on=row["worked_on"] or "",
        started=row["started"] or "",
        ended=row["ended"] or "",
        created=row["created"] or "",
    )


def _insert(settings: Settings, entry: WorkEntry) -> WorkEntry:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.insert_work_entry(settings, entry)
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO work_log"
            " (id, project, minutes, note, worked_on, started, ended, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.id, entry.project, entry.minutes, entry.note, entry.worked_on,
             entry.started, entry.ended, entry.created),
        )
    return entry


def _finish(settings: Settings, entry_id: str, ended: str, minutes: int, note: str) -> None:
    if storage_postgres := postgres_backend(settings):
        storage_postgres.finish_work_entry(settings, entry_id, ended, minutes, note)
        return
    with _connect(settings) as conn:
        conn.execute(
            "UPDATE work_log SET ended = ?, minutes = ?, note = ? WHERE id = ?",
            (ended, minutes, note, entry_id),
        )


def get_entry(settings: Settings, entry_id: str) -> WorkEntry | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_work_entry(settings, entry_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM work_log WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_entry(row) if row else None


def list_entries(settings: Settings, project: str = "") -> list[WorkEntry]:
    """Entries newest-first (by date, then created). ``project`` filters by name
    (case-insensitive exact match); empty returns all."""
    if storage_postgres := postgres_backend(settings):
        entries = storage_postgres.list_work_entries(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM work_log").fetchall()
        entries = [_row_to_entry(r) for r in rows]
    needle = project.strip().lower()
    if needle:
        entries = [e for e in entries if e.project.lower() == needle]
    return sorted(entries, key=lambda e: (e.worked_on, e.created), reverse=True)


def delete_entry(settings: Settings, entry_id: str) -> WorkEntry | None:
    """Delete one entry by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_work_entry(settings, entry_id)
    existing = get_entry(settings, entry_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM work_log WHERE id = ?", (entry_id,))
    return existing


def running_entry(settings: Settings) -> WorkEntry | None:
    """The running timer (started, not yet ended), or ``None``."""
    for entry in list_entries(settings):
        if entry.started and not entry.ended:
            return entry
    return None


def elapsed_minutes(settings: Settings, entry: WorkEntry) -> int:
    """Whole minutes a running timer has been going (0 on a bad stamp)."""
    started = parse_dt(entry.started)
    if started is None:
        return 0
    from ..calendar.context import now

    return max(0, round((now(settings) - started).total_seconds() / 60))


def stop_entry(settings: Settings, note: str = "") -> WorkEntry | None:
    """Stop the running timer, recording its duration; ``None`` when none runs.

    The duration is wall-clock start→stop, floored at one minute so an
    immediate stop still leaves a visible trace to remove or keep. A given
    ``note`` is appended to whatever the start noted.
    """
    running = running_entry(settings)
    if running is None:
        return None
    minutes = max(1, elapsed_minutes(settings, running))
    merged = "; ".join(part for part in (running.note, note.strip()) if part)
    ended = _stamp_now(settings)
    _finish(settings, running.id, ended, minutes, merged)
    return get_entry(settings, running.id)


def start_entry(
    settings: Settings, project: str, note: str = ""
) -> tuple[WorkEntry, WorkEntry | None]:
    """Start the clock on ``project``, stopping any running timer first.

    Returns ``(started, stopped)`` — ``stopped`` is the previously running
    entry (now closed with its duration) or ``None``, so the caller can report
    the switch in one breath.
    """
    stopped = stop_entry(settings)
    now = _stamp_now(settings)
    entry = WorkEntry(
        id=uuid.uuid4().hex[:12],
        project=project.strip(),
        minutes=0,
        note=note.strip(),
        worked_on=_today(settings).isoformat(),
        started=now,
        ended="",
        created=now,
    )
    return _insert(settings, entry), stopped


def log_entry(
    settings: Settings, project: str, minutes: object, note: str = "", on: str = ""
) -> WorkEntry | None:
    """Append a finished stretch of work directly ("2h on X yesterday").

    ``None`` when the project is blank or the minutes aren't positive.
    ``on`` defaults to today.
    """
    name = project.strip()
    duration = coerce_minutes(minutes)
    if not name or duration <= 0:
        return None
    worked_on = parse_date(on) or _today(settings)
    entry = WorkEntry(
        id=uuid.uuid4().hex[:12],
        project=name,
        minutes=duration,
        note=note.strip(),
        worked_on=worked_on.isoformat(),
        created=_stamp_now(settings),
    )
    return _insert(settings, entry)


def project_names(settings: Settings) -> list[str]:
    """Distinct project names (case-insensitive), most-recently-worked first."""
    seen: dict[str, str] = {}
    for entry in list_entries(settings):  # already newest-first
        seen.setdefault(entry.project.lower(), entry.project)
    return list(seen.values())
