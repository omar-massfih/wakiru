"""Registry of live conversation threads — who the assistant talks to, where.

Every channel funnels its turns through ``chat.run_upkeep``, which touches
this registry, so it accrues the set of real conversations (Telegram chats,
Slack threads, web sessions, the CLI) without any per-channel wiring. Two
things read it:

* Proactive loop-in (:mod:`assistant.proactive`) — which Slack threads saw a
  broadcast push, so "what was that reminder about?" works there too.
* The heartbeat — ``last_contact`` gives "how long since we last spoke", one
  of its wake triggers.

SQLite-only (``memory/threads.db``, created lazily): thread liveness is
derived data — losing it costs a few pushes' loop-in, nothing durable — so
the Postgres backend intentionally has no mirror yet.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings, postgres_backend
from .sqlite_util import open_db, transaction

logger = logging.getLogger(__name__)

_KNOWN_CHANNELS = frozenset({"telegram", "slack", "cli"})


@dataclass(frozen=True)
class ThreadInfo:
    thread_id: str
    channel: str
    last_user_at: str
    last_assistant_at: str


def channel_of(thread_id: str) -> str:
    """The channel a thread id belongs to; bare uuids are the HTTP/web channel."""
    prefix = thread_id.split(":", 1)[0]
    return prefix if prefix in _KNOWN_CHANNELS else "http"


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.threads_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS known_threads ("
        " thread_id TEXT PRIMARY KEY, channel TEXT NOT NULL,"
        " last_user_at TEXT NOT NULL DEFAULT '',"
        " last_assistant_at TEXT NOT NULL DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def touch(
    settings: Settings, thread_id: str, *, user: bool = True, assistant: bool = True
) -> None:
    """Record activity on a thread (upsert). Best-effort by design at call sites."""
    if not thread_id:
        return
    stamp = now(settings).isoformat(timespec="seconds")
    if storage_postgres := postgres_backend(settings):
        storage_postgres.touch_thread(
            settings, thread_id, channel_of(thread_id), stamp, user, assistant
        )
        return
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO known_threads (thread_id, channel, last_user_at, last_assistant_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(thread_id) DO UPDATE SET"
            "  last_user_at = CASE WHEN ? THEN excluded.last_user_at ELSE last_user_at END,"
            "  last_assistant_at = CASE WHEN ? THEN excluded.last_assistant_at"
            "   ELSE last_assistant_at END",
            (
                thread_id,
                channel_of(thread_id),
                stamp if user else "",
                stamp if assistant else "",
                user,
                assistant,
            ),
        )


def known_threads(settings: Settings, channel: str | None = None) -> list[ThreadInfo]:
    """Every registered thread, optionally filtered to one channel."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.known_threads(settings, channel)
    query = "SELECT * FROM known_threads"
    params: tuple = ()
    if channel:
        query += " WHERE channel = ?"
        params = (channel,)
    with _connect(settings) as conn:
        return [
            ThreadInfo(
                thread_id=row["thread_id"],
                channel=row["channel"],
                last_user_at=row["last_user_at"],
                last_assistant_at=row["last_assistant_at"],
            )
            for row in conn.execute(query, params)
        ]


def last_contact(settings: Settings) -> datetime | None:
    """When the user last spoke to the assistant, on any channel."""
    stamps = [
        parsed
        for info in known_threads(settings)
        if info.last_user_at and (parsed := parse_dt(info.last_user_at)) is not None
    ]
    return max(stamps, default=None)
