"""Proactive reminders — the calendar's wall-clock output path.

The read path (:mod:`.context`) and write path (:mod:`.ops`) both only run when the
user chats. Reminders are the missing third path: unprompted nudges ahead of an
event ("Dentist in 1 hour"), driven by a wall-clock ticker rather than chat traffic.

:func:`run_reminders` is the entry point. On each call it finds events entering a
configured *lead* window (:attr:`Settings.reminder_lead_minutes`), fires each one
exactly once via a small SQLite dedupe ledger, and pushes it through
:func:`assistant.notify.deliver_webhook`. It is best-effort and idempotent, so the
in-process ticker and a manual ``POST /reminders/run`` can both drive it safely.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta

from ..config import Settings, get_settings
from ..notify import deliver_webhook
from . import store
from .context import now

logger = logging.getLogger(__name__)

# Fired-reminder rows older than this are pruned on each run so the ledger, which
# only ever grows, stays bounded without any separate maintenance job.
_LEDGER_RETENTION_DAYS = 30


def _connect(settings: Settings) -> sqlite3.Connection:
    """Open the calendar DB and ensure the dedupe ledger exists.

    Mirrors :func:`assistant.calendar.store._connect` (WAL + busy timeout, a fresh
    connection per operation) and shares the same ``calendar.db`` file.
    """
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.calendar_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reminders_fired ("
        " event_id TEXT NOT NULL, event_start TEXT NOT NULL,"
        " lead_minutes INTEGER NOT NULL, fired_at TEXT NOT NULL,"
        " PRIMARY KEY (event_id, event_start, lead_minutes))"
    )
    return conn


def _humanize(delta: timedelta) -> str:
    """Render a positive time-until as a short phrase: 'in 30 min' / 'in 1 hour'."""
    minutes = round(delta.total_seconds() / 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"in {minutes} min"
    if minutes < 1440:
        hours = round(minutes / 60)
        return f"in {hours} hour{'s' if hours != 1 else ''}"
    days = round(minutes / 1440)
    return f"in {days} day{'s' if days != 1 else ''}"


def due_reminders(settings: Settings, current: datetime | None = None) -> list[dict]:
    """Reminders that should fire as of ``current`` (defaults to the assistant's now).

    For every upcoming event and every configured lead L, the event is due when it
    starts within the next L minutes (and is not already past). Returns one dict per
    (event, lead): ``{event_id, title, start, lead_minutes, message}``. Pure — it
    does not touch the ledger or deliver anything.
    """
    leads = settings.reminder_lead_minutes
    if not leads:
        return []

    current = current or now(settings)
    horizon = current + timedelta(minutes=max(leads))
    events = store.list_events(settings, start_from=current, start_to=horizon)

    reminders: list[dict] = []
    for event in events:
        start = store.parse_dt(event.start)
        if start is None:
            continue
        remaining = start - current
        for lead in leads:
            if timedelta(0) <= remaining <= timedelta(minutes=lead):
                reminders.append(
                    {
                        "event_id": event.id,
                        "title": event.title,
                        "start": event.start,
                        "lead_minutes": lead,
                        "message": f"{event.title} {_humanize(remaining)}",
                    }
                )
    return reminders


def _prune_ledger(conn: sqlite3.Connection, current: datetime) -> None:
    cutoff = (current - timedelta(days=_LEDGER_RETENTION_DAYS)).isoformat()
    conn.execute("DELETE FROM reminders_fired WHERE fired_at < ?", (cutoff,))


def run_reminders(settings: Settings | None = None) -> list[dict]:
    """Fire every reminder now due, exactly once, and return what was sent.

    Best-effort and idempotent: each due reminder is claimed with an atomic
    ``INSERT OR IGNORE`` on the ledger, so a reminder already fired (by an earlier
    tick or an overlapping manual call) is silently skipped. A rescheduled event
    fires afresh because the ledger key includes the event's start. No-op returning
    ``[]`` when ``enable_reminders`` is false.
    """
    settings = settings or get_settings()
    if not settings.enable_reminders:
        return []

    current = now(settings)
    fired_at = current.isoformat(timespec="seconds")
    # Compute the due list first, with its own (store) connections, so the ledger
    # write transaction below never overlaps a nested connection to the same DB.
    due = due_reminders(settings, current)

    sent: list[dict] = []
    with _connect(settings) as conn:
        _prune_ledger(conn, current)
        for reminder in due:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO reminders_fired"
                " (event_id, event_start, lead_minutes, fired_at) VALUES (?, ?, ?, ?)",
                (
                    reminder["event_id"],
                    reminder["start"],
                    reminder["lead_minutes"],
                    fired_at,
                ),
            )
            if cursor.rowcount != 1:
                continue  # already fired for this (event, start, lead)
            deliver_webhook(settings, reminder)
            sent.append(reminder)

    if sent:
        logger.info("fired %d reminder(s): %s", len(sent), "; ".join(r["message"] for r in sent))
    return sent
