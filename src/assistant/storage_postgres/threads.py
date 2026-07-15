"""Live-thread registry for the Postgres backend.

Mirrors :mod:`assistant.threads` so the "who does the assistant talk to, and when
did we last speak" registry (the heartbeat's contact-staleness signal and the
Slack loop-in set) is durable in Neon instead of a prod-only ``threads.db``.
"""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect


def ensure_threads_schema(settings: Settings) -> None:
    if _schema_done(settings, "threads"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_threads (
              thread_id TEXT PRIMARY KEY,
              channel TEXT NOT NULL,
              last_user_at TEXT NOT NULL DEFAULT '',
              last_assistant_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "threads")


def touch_thread(
    settings: Settings,
    thread_id: str,
    channel: str,
    stamp: str,
    user: bool,
    assistant: bool,
) -> None:
    ensure_threads_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_threads"
            " (thread_id, channel, last_user_at, last_assistant_at)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (thread_id) DO UPDATE SET"
            "  last_user_at = CASE WHEN %s THEN EXCLUDED.last_user_at"
            "   ELSE assistant_threads.last_user_at END,"
            "  last_assistant_at = CASE WHEN %s THEN EXCLUDED.last_assistant_at"
            "   ELSE assistant_threads.last_assistant_at END",
            (
                thread_id,
                channel,
                stamp if user else "",
                stamp if assistant else "",
                user,
                assistant,
            ),
        )


def known_threads(settings: Settings, channel: str | None = None) -> list:
    from ..threads import ThreadInfo

    ensure_threads_schema(settings)
    query = "SELECT * FROM assistant_threads"
    params: tuple = ()
    if channel:
        query += " WHERE channel = %s"
        params = (channel,)
    with connect(settings) as conn:
        rows = _rows(conn.execute(query, params))
    return [
        ThreadInfo(
            thread_id=str(row["thread_id"]),
            channel=str(row["channel"]),
            last_user_at=str(row.get("last_user_at") or ""),
            last_assistant_at=str(row.get("last_assistant_at") or ""),
        )
        for row in rows
    ]
