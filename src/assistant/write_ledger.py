"""Generic write-ledger/undo machinery shared by the calendar and tasks.

Each subsystem logs every applied write to a ``write_log`` table in its own
SQLite DB (or Postgres via :mod:`assistant.storage_postgres`), grouped per turn
by a ``batch_id``, so replying "undo" reverts exactly what one turn changed —
deterministically, no LLM call. The subsystem supplies a :class:`LedgerSpec`
naming its DB, target column, and Postgres adapters, plus a revert callback;
keeping the driver here means the two ledgers cannot drift apart (a copy-paste
divergence in the tasks twin once silently broke Postgres undo).

The undo discipline is: compute the batch first, mutate the store second,
claim the ledger rows third — no SQLite connection is held open across the
``store.*`` mutations (matching :func:`assistant.calendar.reminders.run_reminders`).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from .config import Settings, postgres_backend
from .sqlite_util import open_db, transaction

logger = logging.getLogger(__name__)

# A revert callback: apply the reverse of one logged write, returning a short
# user-facing summary, or None when there was nothing to revert.
RevertRow = Callable[[Settings, dict], str | None]


class LedgerSpec:
    """Where one subsystem's write ledger lives and how to reach its backend."""

    def __init__(
        self,
        kind: str,
        db_path: Callable[[Settings], Path],
        target_column: str,
        pg_record: str,
        pg_rows: str,
        pg_mark_undone: str,
    ) -> None:
        self.kind = kind  # for log messages: "calendar" / "task"
        self.db_path = db_path
        self.target_column = target_column  # "event_id" / "task_id"
        # Names of the storage_postgres adapters, resolved at call time so test
        # monkeypatches on that module keep working.
        self.pg_record = pg_record
        self.pg_rows = pg_rows
        self.pg_mark_undone = pg_mark_undone


def _now(settings: Settings) -> datetime:
    # Late import: calendar/__init__ imports .undo which imports this module.
    from .calendar.context import now

    return now(settings)


def _parse_dt(value: str) -> datetime | None:
    from .calendar.store import parse_dt

    return parse_dt(value)


def _open(spec: LedgerSpec, settings: Settings) -> sqlite3.Connection:
    conn = open_db(spec.db_path(settings))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS write_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
        f" batch_id TEXT NOT NULL, {spec.target_column} TEXT NOT NULL, op TEXT NOT NULL,"
        " summary TEXT NOT NULL, before_json TEXT, applied_at TEXT NOT NULL,"
        " undone_at TEXT)"
    )
    return conn


@contextmanager
def connect(spec: LedgerSpec, settings: Settings) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see store._connect)."""
    with transaction(_open(spec, settings)) as conn:
        yield conn


def record_write(
    spec: LedgerSpec,
    settings: Settings,
    thread_id: str,
    batch_id: str,
    target_id: str,
    op: str,
    summary: str,
    before_json: str | None,
) -> None:
    """Log one applied mutation so it can later be undone. No-op without a thread."""
    if not thread_id:
        return
    try:
        applied_at = _now(settings).isoformat(timespec="seconds")
        if storage_postgres := postgres_backend(settings):
            record = getattr(storage_postgres, spec.pg_record)
            record(
                settings, thread_id, batch_id, target_id, op, summary,
                before_json, applied_at,
            )
            return
        with connect(spec, settings) as conn:
            conn.execute(
                "INSERT INTO write_log"
                f" (thread_id, batch_id, {spec.target_column}, op, summary,"
                " before_json, applied_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, batch_id, target_id, op, summary, before_json, applied_at),
            )
    except Exception:
        logger.exception(
            "failed to record %s undo log for %s (thread %s)",
            spec.kind, target_id, thread_id,
        )


def latest_applied_at(
    spec: LedgerSpec, settings: Settings, thread_id: str, window_minutes: int
) -> datetime | None:
    """Timestamp of the most recent still-undoable write on ``thread_id`` (or
    ``None`` if nothing is recent enough). Lets the cross-ledger arbiter in
    :mod:`assistant.undo` decide which subsystem owns the most recent write."""
    cutoff = _now(settings) - timedelta(minutes=window_minutes)
    if storage_postgres := postgres_backend(settings):
        rows = getattr(storage_postgres, spec.pg_rows)(settings, thread_id)
        if not rows:
            return None
        applied_at = _parse_dt(str(rows[0]["applied_at"]))
        if applied_at is None or applied_at < cutoff:
            return None
        return applied_at
    with connect(spec, settings) as conn:
        latest = conn.execute(
            "SELECT applied_at FROM write_log WHERE thread_id = ? AND undone_at IS NULL"
            " ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    if latest is None:
        return None
    applied_at = _parse_dt(latest["applied_at"])
    if applied_at is None or applied_at < cutoff:
        return None
    return applied_at


def undo_latest(
    spec: LedgerSpec,
    settings: Settings,
    thread_id: str,
    window_minutes: int,
    revert_row: RevertRow,
) -> str:
    """Revert the most recent undoable batch of writes on ``thread_id``.

    Reverts every row sharing the latest non-undone row's ``batch_id`` (a turn
    can apply several ops), oldest-mutation-last (``id DESC``).
    """
    cutoff = _now(settings) - timedelta(minutes=window_minutes)

    if storage_postgres := postgres_backend(settings):
        all_rows = getattr(storage_postgres, spec.pg_rows)(settings, thread_id)
        if not all_rows:
            return "Nothing to undo."
        latest = all_rows[0]
        applied_at = _parse_dt(str(latest["applied_at"]))
        if applied_at is None or applied_at < cutoff:
            return "Nothing recent enough to undo."
        rows = [r for r in all_rows if r["batch_id"] == latest["batch_id"]]
    else:
        with connect(spec, settings) as conn:
            latest = conn.execute(
                "SELECT * FROM write_log WHERE thread_id = ? AND undone_at IS NULL"
                " ORDER BY id DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
            if latest is None:
                return "Nothing to undo."
            # Compare as datetimes, not ISO strings: stamps written under different
            # UTC offsets (a DST change) don't order lexically.
            applied_at = _parse_dt(latest["applied_at"])
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
        summary = revert_row(settings, dict(row))
        if summary is not None:
            summaries.append(summary)
            reverted_ids.append(row["id"])

    if not reverted_ids:
        return "Nothing to undo."

    undone_at = _now(settings).isoformat(timespec="seconds")
    if storage_postgres := postgres_backend(settings):
        getattr(storage_postgres, spec.pg_mark_undone)(
            settings, reverted_ids, undone_at
        )
    else:
        with connect(spec, settings) as conn:
            conn.executemany(
                "UPDATE write_log SET undone_at = ? WHERE id = ?",
                [(undone_at, rid) for rid in reverted_ids],
            )

    return "Undone: " + "; ".join(summaries)
