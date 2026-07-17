"""Undo ledger for the to-do list — the tasks confirmation/safety-net path.

A direct parallel to :mod:`assistant.calendar.undo`: every applied task write
(add/complete/update/remove) is logged, grouped per turn by a ``batch_id``, so
replying "undo" reverts exactly what one turn changed — deterministically, no
LLM call. The chat layer arbitrates between this ledger and the calendar's so
"undo" reverts the most recent write of either kind (see
:func:`assistant.undo.undo_latest`). The generic driver lives in
:mod:`assistant.write_ledger`; this module supplies the tasks spec and the
task-specific revert.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime

from .. import write_ledger
from ..config import Settings
from . import store

logger = logging.getLogger(__name__)

_SPEC = write_ledger.LedgerSpec(
    kind="task",
    db_path=lambda settings: settings.tasks_db_path,
    target_column="task_id",
    pg_record="record_task_write",
    pg_rows="task_write_rows",
    pg_mark_undone="mark_task_writes_undone",
)


def _connect(settings: Settings):
    """One transaction on this ledger's local DB (see write_ledger.connect)."""
    return write_ledger.connect(_SPEC, settings)


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
    write_ledger.record_write(
        _SPEC, settings, thread_id, batch_id, task_id, op, summary,
        json.dumps(asdict(before)) if before else None,
    )


def completed_in_batch(
    settings: Settings, thread_id: str, batch_id: str, task_id: str
) -> bool:
    """True when this turn's batch already completed ``task_id`` (see
    write_ledger.batch_has; keeps a recurring task's roll-forward idempotent
    within a turn)."""
    return write_ledger.batch_has(
        _SPEC, settings, thread_id, batch_id, task_id, "complete"
    )


def latest_applied_at(
    settings: Settings, thread_id: str, window_minutes: int
) -> datetime | None:
    """The timestamp of the most recent still-undoable write on ``thread_id``,
    or ``None`` if there's nothing recent enough. Used by the cross-ledger
    arbiter to decide which subsystem's "undo" wins."""
    return write_ledger.latest_applied_at(_SPEC, settings, thread_id, window_minutes)


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
    """Revert the most recent undoable batch of task writes on ``thread_id``."""
    return write_ledger.undo_latest(
        _SPEC, settings, thread_id, window_minutes, _revert_row
    )
