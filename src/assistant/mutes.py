"""Reminder mutes — the shared quiet switch between the agent and the tickers.

The reminder tickers (:mod:`assistant.calendar.reminders`,
:mod:`assistant.tasks.reminders`, :mod:`assistant.briefing`) are wall-clock
loops that never see the conversation, so "I'm sick, skipping exercise" said in
chat used to change nothing — the nudges kept firing. This module is the bridge:
the conversational agent writes mutes through its tools
(:func:`assistant.tools` ``mute_reminders``/``unmute_reminders``) and every
ticker filters its due list against them *before claiming*, so a mute that
expires mid-window lets the remaining nudges fire normally.

A mute silences delivery only; it never touches the calendar or tasks. Scopes:
``event`` / ``task`` target one item by id, ``all`` holds every push (including
the daily briefing). Rows expire at ``until`` and are pruned on write, mirroring
the fired-ledger self-maintenance in the reminder modules.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from .calendar.store import parse_dt
from .config import Settings, postgres_backend
from .sqlite_util import open_db, transaction

logger = logging.getLogger(__name__)

# The id key a due-reminder dict carries, per mute scope (see due_reminders /
# due_task_reminders — the dict shapes differ only in this key).
_ID_KEYS = {"event": "event_id", "task": "task_id"}


def _open(settings: Settings) -> sqlite3.Connection:
    """Open the mutes DB and ensure the table exists (WAL, fresh connection —
    the same discipline as calendar.reminders._open)."""
    conn = open_db(settings.mutes_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reminder_mutes ("
        " scope TEXT NOT NULL, target_id TEXT NOT NULL,"
        " until TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',"
        " created_at TEXT NOT NULL,"
        " PRIMARY KEY (scope, target_id))"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def _prune(conn: sqlite3.Connection, current: datetime) -> None:
    """Drop expired (and unparseable) mutes. Compared as datetimes in Python,
    not ISO strings in SQL — stamps under different UTC offsets don't order
    lexically (same rationale as the fired-ledger pruning)."""
    stale = [
        (row["scope"], row["target_id"])
        for row in conn.execute("SELECT scope, target_id, until FROM reminder_mutes")
        if (until := parse_dt(row["until"])) is None or until <= current
    ]
    conn.executemany(
        "DELETE FROM reminder_mutes WHERE scope = ? AND target_id = ?", stale
    )


def set_mute(
    settings: Settings,
    scope: str,
    target_id: str,
    until: datetime,
    reason: str = "",
    current: datetime | None = None,
) -> None:
    """Upsert one mute: no nudges for (scope, target_id) until ``until``."""
    current = current or datetime.now().astimezone()
    if storage_postgres := postgres_backend(settings):
        storage_postgres.set_mute(settings, scope, target_id, until, reason, current)
        return
    with _connect(settings) as conn:
        _prune(conn, current)
        conn.execute(
            "INSERT OR REPLACE INTO reminder_mutes"
            " (scope, target_id, until, reason, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                scope,
                target_id,
                until.isoformat(timespec="seconds"),
                reason,
                current.isoformat(timespec="seconds"),
            ),
        )


def clear_mute(settings: Settings, scope: str, target_id: str) -> bool:
    """Delete one mute; True if a row existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.clear_mute(settings, scope, target_id)
    with _connect(settings) as conn:
        cursor = conn.execute(
            "DELETE FROM reminder_mutes WHERE scope = ? AND target_id = ?",
            (scope, target_id),
        )
        return cursor.rowcount > 0


def active_mutes(settings: Settings, current: datetime) -> dict[tuple[str, str], datetime]:
    """Every unexpired mute as ``{(scope, target_id): until}`` — one read per tick."""
    if storage_postgres := postgres_backend(settings):
        rows = storage_postgres.list_mutes(settings)
    else:
        with _connect(settings) as conn:
            rows = [
                (row["scope"], row["target_id"], row["until"])
                for row in conn.execute(
                    "SELECT scope, target_id, until FROM reminder_mutes"
                )
            ]
    return {
        (scope, target_id): until
        for scope, target_id, raw in rows
        if (until := parse_dt(str(raw))) is not None and until > current
    }


def all_muted(settings: Settings, current: datetime) -> bool:
    """True while an ``all`` mute is active (holds every proactive push)."""
    return ("all", "") in active_mutes(settings, current)


def filter_muted(
    settings: Settings, due: list[dict], current: datetime, scope: str
) -> list[dict]:
    """Drop due reminders silenced by an active mute (per-item or ``all``).

    Called before the fired-ledger claim, so muted bands are never claimed and
    delivery resumes on the first tick after the mute expires.
    """
    if not due:
        return due
    mutes = active_mutes(settings, current)
    if not mutes:
        return due
    if ("all", "") in mutes:
        logger.info("all reminders muted; holding %d due reminder(s)", len(due))
        return []
    id_key = _ID_KEYS[scope]
    kept = [r for r in due if (scope, str(r[id_key])) not in mutes]
    if len(kept) < len(due):
        logger.info("muted %d of %d due %s reminder(s)", len(due) - len(kept), len(due), scope)
    return kept
