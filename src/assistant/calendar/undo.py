"""Undo ledger — the calendar's confirmation/safety-net path.

Every applied write (create/reschedule/cancel/skip/move) is logged here,
grouped per turn by a ``batch_id`` (see :mod:`.ops`), so replying "undo"
reverts exactly what one turn changed — deterministically, with no LLM call
involved. Mirrors :mod:`.reminders`: its own small SQLite table sharing
``calendar.db``, a fresh connection per operation (WAL + busy timeout).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import timedelta

from ..config import Settings
from . import store
from .context import now

logger = logging.getLogger(__name__)


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.calendar_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS write_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
        " batch_id TEXT NOT NULL, event_id TEXT NOT NULL, op TEXT NOT NULL,"
        " summary TEXT NOT NULL, before_json TEXT, applied_at TEXT NOT NULL,"
        " undone_at TEXT)"
    )
    return conn


def record_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    event_id: str,
    op: str,
    summary: str,
    before: store.Event | None,
) -> None:
    """Log one applied mutation so it can later be undone. No-op without a thread."""
    if not thread_id:
        return
    try:
        with _connect(settings) as conn:
            conn.execute(
                "INSERT INTO write_log"
                " (thread_id, batch_id, event_id, op, summary, before_json, applied_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id, batch_id, event_id, op, summary,
                    json.dumps(asdict(before)) if before else None,
                    now(settings).isoformat(timespec="seconds"),
                ),
            )
    except Exception:
        logger.exception("failed to record undo log for %s (thread %s)", event_id, thread_id)


def _revert_row(settings: Settings, row: sqlite3.Row) -> str | None:
    """Apply the reverse of one logged write; return a short summary, or None on failure."""
    try:
        if row["op"] == "create":
            deleted = store.delete_event(settings, row["event_id"])
            return f"removed: {deleted.title}" if deleted else None
        if not row["before_json"]:
            return None
        before = store.Event(**json.loads(row["before_json"]))
        restored = store.restore_event(settings, before)
        return f"restored: {restored.title} @ {restored.start}"
    except Exception:
        logger.exception("failed to revert write_log row %s", row["id"])
        return None


def undo_latest(settings: Settings, thread_id: str, window_minutes: int) -> str:
    """Revert the most recent undoable batch of writes on ``thread_id``.

    Reverts every row sharing the latest non-undone row's ``batch_id`` (a turn
    can apply several ops), oldest-mutation-last (``id DESC``). No SQLite
    connection is held open across the ``store.*`` mutations, matching the
    "compute first, mutate second, claim third" discipline used by
    :func:`assistant.calendar.reminders.run_reminders`.
    """
    cutoff = (now(settings) - timedelta(minutes=window_minutes)).isoformat(timespec="seconds")

    with _connect(settings) as conn:
        latest = conn.execute(
            "SELECT * FROM write_log WHERE thread_id = ? AND undone_at IS NULL"
            " ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if latest is None:
            return "Nothing to undo."
        if latest["applied_at"] < cutoff:
            return "Nothing recent enough to undo."
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM write_log WHERE thread_id = ? AND batch_id = ?"
                " AND undone_at IS NULL ORDER BY id DESC",
                (thread_id, latest["batch_id"]),
            ).fetchall()
        ]

    summaries: list[str] = []
    reverted_ids: list[int] = []
    for row in rows:
        summary = _revert_row(settings, row)
        if summary is not None:
            summaries.append(summary)
            reverted_ids.append(row["id"])

    if not reverted_ids:
        return "Nothing to undo."

    undone_at = now(settings).isoformat(timespec="seconds")
    with _connect(settings) as conn:
        conn.executemany(
            "UPDATE write_log SET undone_at = ? WHERE id = ?",
            [(undone_at, rid) for rid in reverted_ids],
        )

    return "Undone: " + "; ".join(summaries)
