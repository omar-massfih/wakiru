"""Undo ledger for the to-do list — the tasks confirmation/safety-net path.

A direct parallel to :mod:`assistant.calendar.undo`: every applied task write
(add/complete/update/remove) is logged to a ``write_log`` table in ``tasks.db``,
grouped per turn by a ``batch_id``, so replying "undo" reverts exactly what one
turn changed — deterministically, no LLM call. The chat layer arbitrates between
this ledger and the calendar's so "undo" reverts the most recent write of either
kind (see :func:`assistant.undo.undo_latest`).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta

from ..calendar.context import now
from ..calendar.store import parse_dt
from ..config import Settings
from . import store

logger = logging.getLogger(__name__)


def _open(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.tasks_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS write_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
        " batch_id TEXT NOT NULL, task_id TEXT NOT NULL, op TEXT NOT NULL,"
        " summary TEXT NOT NULL, before_json TEXT, applied_at TEXT NOT NULL,"
        " undone_at TEXT)"
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


def record_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    task_id: str,
    op: str,
    summary: str,
    before: store.Task | None,
) -> None:
    """Log one applied task mutation so it can later be undone. No-op without a thread."""
    if not thread_id:
        return
    try:
        if settings.storage_backend == "postgres":
            from .. import storage_postgres

            storage_postgres.record_task_write(
                settings, thread_id, batch_id, task_id, op, summary,
                json.dumps(asdict(before)) if before else None,
                now(settings).isoformat(timespec="seconds"),
            )
            return
        with _connect(settings) as conn:
            conn.execute(
                "INSERT INTO write_log"
                " (thread_id, batch_id, task_id, op, summary, before_json, applied_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id, batch_id, task_id, op, summary,
                    json.dumps(asdict(before)) if before else None,
                    now(settings).isoformat(timespec="seconds"),
                ),
            )
    except Exception:
        logger.exception("failed to record task undo log for %s (thread %s)", task_id, thread_id)


def latest_applied_at(settings: Settings, thread_id: str, window_minutes: int) -> datetime | None:
    """The timestamp of the most recent still-undoable write on ``thread_id``,
    or ``None`` if there's nothing recent enough. Used by the cross-ledger
    arbiter to decide which subsystem's "undo" wins."""
    cutoff = now(settings) - timedelta(minutes=window_minutes)
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        rows = storage_postgres.task_write_rows(settings, thread_id)
        if not rows:
            return None
        applied_at = parse_dt(str(rows[0]["applied_at"]))
        if applied_at is None or applied_at < cutoff:
            return None
        return applied_at
    with _connect(settings) as conn:
        latest = conn.execute(
            "SELECT applied_at FROM write_log WHERE thread_id = ? AND undone_at IS NULL"
            " ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    if latest is None:
        return None
    applied_at = parse_dt(latest["applied_at"])
    if applied_at is None or applied_at < cutoff:
        return None
    return applied_at


def _revert_row(settings: Settings, row: dict) -> str | None:
    """Apply the reverse of one logged task write; return a short summary, or None."""
    try:
        if row["op"] == "add":
            deleted = store.delete_task(settings, row["task_id"])
            return f"removed: {deleted.title}" if deleted else None
        if not row["before_json"]:
            return None
        before = store.Task(**json.loads(row["before_json"]))
        restored = store.restore_task(settings, before)
        return f"restored: {restored.title}"
    except Exception:
        logger.exception("failed to revert task write_log row %s", row["id"])
        return None


def undo_latest(settings: Settings, thread_id: str, window_minutes: int) -> str:
    """Revert the most recent undoable batch of task writes on ``thread_id``.

    Same discipline as :func:`assistant.calendar.undo.undo_latest`: compute the
    batch first, mutate the store second, claim the ledger rows third.
    """
    cutoff = now(settings) - timedelta(minutes=window_minutes)
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        all_rows = storage_postgres.task_write_rows(settings, thread_id)
        if not all_rows:
            return "Nothing to undo."
        latest = all_rows[0]
        applied_at = parse_dt(str(latest["applied_at"]))
        if applied_at is None or applied_at < cutoff:
            return "Nothing recent enough to undo."
        rows = [r for r in all_rows if r["batch_id"] == latest["batch_id"]]
    else:
        with _connect(settings) as conn:
            latest = conn.execute(
                "SELECT * FROM write_log WHERE thread_id = ? AND undone_at IS NULL"
                " ORDER BY id DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
            if latest is None:
                return "Nothing to undo."
            applied_at = parse_dt(latest["applied_at"])
            if applied_at is None or applied_at < cutoff:
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
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        storage_postgres.mark_task_writes_undone(settings, reverted_ids, undone_at)
    else:
        with _connect(settings) as conn:
            conn.executemany(
                "UPDATE write_log SET undone_at = ? WHERE id = ?",
                [(undone_at, rid) for rid in reverted_ids],
            )
    return "Undone: " + "; ".join(summaries)
