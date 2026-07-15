"""Durable push queue for CalDAV writes that didn't reach the server.

A local calendar write always lands first; the CalDAV push is a best-effort
second step (see :func:`assistant.calendar.ops._push_caldav`). When that push
fails — the server is down, or an ``If-Match`` precondition lost a race — the
intent is parked here so the background reconcile (:mod:`assistant.api`) can
retry it. The queue lives in its own table so it survives the local row's
deletion: a failed *cancel* removes the event locally but the remote ``DELETE``
still has to happen, and the href/etag/ical snapshot needed to retry is here.

One pending entry per event (keyed by ``event_id``) — the latest intent wins,
which is what a retry wants: re-enqueuing a create-then-reschedule should push
the final state, not replay both.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from ..config import Settings

# The two remote operations a queued push maps to.
OP_PUT = "put"
OP_DELETE = "delete"


def _open(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.calendar_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS caldav_outbox ("
        " event_id TEXT PRIMARY KEY, op TEXT NOT NULL,"
        " href TEXT DEFAULT '', etag TEXT DEFAULT '', ical TEXT DEFAULT '',"
        " queued_at TEXT NOT NULL)"
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


def _now(settings: Settings) -> str:
    from .context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def enqueue(
    settings: Settings,
    event_id: str,
    op: str,
    *,
    href: str = "",
    etag: str = "",
    ical: str = "",
) -> None:
    """Park a failed push for reconcile; the latest intent per event replaces any prior."""
    queued_at = _now(settings)
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        storage_postgres.caldav_outbox_enqueue(
            settings, event_id, op, href, etag, ical, queued_at
        )
        return
    with _connect(settings) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO caldav_outbox"
            " (event_id, op, href, etag, ical, queued_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, op, href, etag, ical, queued_at),
        )


def pending(settings: Settings) -> list[dict]:
    """Every queued push, oldest first (as dicts: event_id, op, href, etag, ical)."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.caldav_outbox_pending(settings)
    with _connect(settings) as conn:
        rows = conn.execute(
            "SELECT event_id, op, href, etag, ical, queued_at"
            " FROM caldav_outbox ORDER BY queued_at"
        ).fetchall()
    return [dict(row) for row in rows]


def has_pending(settings: Settings, event_id: str) -> bool:
    """Whether a push is queued for this event (the pull must not clobber it)."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return any(r["event_id"] == event_id for r in storage_postgres.caldav_outbox_pending(settings))
    with _connect(settings) as conn:
        row = conn.execute(
            "SELECT 1 FROM caldav_outbox WHERE event_id = ?", (event_id,)
        ).fetchone()
    return row is not None


def clear(settings: Settings, event_id: str) -> None:
    """Drop a queued push once it has been successfully replayed (or abandoned)."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        storage_postgres.caldav_outbox_clear(settings, event_id)
        return
    with _connect(settings) as conn:
        conn.execute("DELETE FROM caldav_outbox WHERE event_id = ?", (event_id,))
