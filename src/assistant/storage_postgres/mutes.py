"""Reminder-mute table for the Postgres backend."""

from __future__ import annotations

from ..config import Settings
from .core import (
    _executemany,
    _rows,
    _schema_done,
    _schema_mark,
    connect,
)


def ensure_mutes_schema(settings: Settings) -> None:
    if _schema_done(settings, "mutes"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_reminder_mutes (
              scope TEXT NOT NULL,
              target_id TEXT NOT NULL,
              until TEXT NOT NULL,
              reason TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              PRIMARY KEY (scope, target_id)
            )
            """
        )
    _schema_mark(settings, "mutes")


def set_mute(settings: Settings, scope: str, target_id: str, until, reason: str, current) -> None:
    from ..calendar.store import parse_dt

    ensure_mutes_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT scope, target_id, until FROM assistant_reminder_mutes"))
        stale = [
            (r["scope"], r["target_id"])
            for r in rows
            if (expiry := parse_dt(str(r["until"]))) is None or expiry <= current
        ]
        _executemany(
            conn,
            "DELETE FROM assistant_reminder_mutes WHERE scope = %s AND target_id = %s",
            stale,
        )
        conn.execute(
            "INSERT INTO assistant_reminder_mutes (scope, target_id, until, reason, created_at)"
            " VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (scope, target_id) DO UPDATE"
            " SET until = EXCLUDED.until, reason = EXCLUDED.reason, created_at = EXCLUDED.created_at",
            (
                scope,
                target_id,
                until.isoformat(timespec="seconds"),
                reason,
                current.isoformat(timespec="seconds"),
            ),
        )


def clear_mute(settings: Settings, scope: str, target_id: str) -> bool:
    ensure_mutes_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "DELETE FROM assistant_reminder_mutes WHERE scope = %s AND target_id = %s"
            " RETURNING scope",
            (scope, target_id),
        )
        return cur.fetchone() is not None


def list_mutes(settings: Settings) -> list[tuple[str, str, str]]:
    ensure_mutes_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT scope, target_id, until FROM assistant_reminder_mutes"))
    return [(str(r["scope"]), str(r["target_id"]), str(r["until"])) for r in rows]
