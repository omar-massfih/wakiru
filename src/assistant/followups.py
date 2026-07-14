"""Followups — the assistant's own initiative store.

A followup is something the assistant decided (or was asked) to come back to
later: "check in tomorrow about the apartment viewing", "ask how the interview
went". Unlike calendar reminders — verbatim nudges fired at exact minutes —
a followup is *deliberative*: when it comes due, the heartbeat
(:mod:`assistant.heartbeat`) wakes the model with the followup's topic and
context and lets it decide what to actually say.

Rows live in ``memory/followups.db`` (SQLite, created lazily; the mutes-store
pattern). Claiming is atomic and claim-first (``UPDATE … WHERE status='open'``),
the same exactly-once discipline as the fired ledgers: a followup is consumed
the moment a heartbeat picks it up, so racing ticks can't double-raise it.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Followup:
    id: str
    due: str
    topic: str
    context: str = ""
    thread_id: str = ""
    created_at: str = ""
    status: str = "open"  # open | fired | cancelled
    fired_at: str = ""


def _open(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.followups_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS followups ("
        " id TEXT PRIMARY KEY, due TEXT NOT NULL, topic TEXT NOT NULL,"
        " context TEXT NOT NULL DEFAULT '', thread_id TEXT NOT NULL DEFAULT '',"
        " created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',"
        " fired_at TEXT NOT NULL DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    conn = _open(settings)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _from_row(row: sqlite3.Row) -> Followup:
    return Followup(**dict(row))


def add(
    settings: Settings,
    due: datetime,
    topic: str,
    context: str = "",
    thread_id: str = "",
) -> Followup:
    """Schedule one followup; returns the stored row."""
    followup = Followup(
        id=uuid.uuid4().hex[:12],
        due=due.isoformat(timespec="seconds"),
        topic=str(topic).strip(),
        context=str(context).strip(),
        thread_id=thread_id,
        created_at=now(settings).isoformat(timespec="seconds"),
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO followups"
            " (id, due, topic, context, thread_id, created_at, status, fired_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'open', '')",
            (
                followup.id,
                followup.due,
                followup.topic,
                followup.context,
                followup.thread_id,
                followup.created_at,
            ),
        )
    return followup


def list_open(settings: Settings) -> list[Followup]:
    """Every open followup, soonest due first."""
    with _connect(settings) as conn:
        rows = conn.execute(
            "SELECT * FROM followups WHERE status = 'open' ORDER BY due"
        ).fetchall()
    return [_from_row(row) for row in rows]


def cancel(settings: Settings, ident: str) -> Followup | None:
    """Cancel one open followup by id or an unambiguous topic reference.

    A topic reference matching more than one open followup is refused rather
    than guessed at — cancelling nothing beats cancelling the wrong intent
    (the same rule the calendar and mute targets follow).
    """
    ident = str(ident).strip()
    if not ident:
        return None
    candidates = list_open(settings)
    matches = [f for f in candidates if f.id == ident]
    if not matches:
        needle = ident.lower()
        matches = [f for f in candidates if needle in f.topic.lower()]
    if len(matches) != 1:
        if len(matches) > 1:
            logger.warning(
                "followup cancel target %r is ambiguous between %d followups; skipping",
                ident,
                len(matches),
            )
        return None
    target = matches[0]
    with _connect(settings) as conn:
        cursor = conn.execute(
            "UPDATE followups SET status = 'cancelled' WHERE id = ? AND status = 'open'",
            (target.id,),
        )
    return target if cursor.rowcount else None


def claim_due(settings: Settings, current: datetime | None = None) -> list[Followup]:
    """Atomically claim every followup now due; each is returned exactly once.

    Claim-first: the status flips to ``fired`` in the same statement that
    selects it, so a racing tick sees nothing left to claim. A claimed followup
    the model then chooses to stay silent about is still consumed — the same
    at-most-once tradeoff the reminder ledgers make.
    """
    current = current or now(settings)
    fired_at = current.isoformat(timespec="seconds")
    due_now = [
        f
        for f in list_open(settings)
        if (due := parse_dt(f.due)) is not None and due <= current
    ]
    claimed: list[Followup] = []
    with _connect(settings) as conn:
        for followup in due_now:
            cursor = conn.execute(
                "UPDATE followups SET status = 'fired', fired_at = ?"
                " WHERE id = ? AND status = 'open'",
                (fired_at, followup.id),
            )
            if cursor.rowcount:
                claimed.append(followup)
    return claimed
