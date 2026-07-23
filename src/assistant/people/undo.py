"""Undo ledger for the people store — the CRM confirmation/safety-net path.

A direct parallel to :mod:`assistant.tasks.undo`: every applied people write
(add/update/log_contact/remove) is logged, grouped per turn by a ``batch_id``,
so replying "undo" reverts exactly what one turn changed — deterministically, no
LLM call. The cross-subsystem arbiter (:func:`assistant.undo.undo_latest`)
consults this ledger alongside the calendar's and the tasks'. The generic driver
lives in :mod:`assistant.write_ledger`; this module supplies the people spec and
the person-specific revert.
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
    kind="person",
    db_path=lambda settings: settings.people_db_path,
    target_column="person_id",
    pg_record="record_person_write",
    pg_rows="person_write_rows",
    pg_mark_undone="mark_person_writes_undone",
)


def record_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    person_id: str,
    op: str,
    summary: str,
    before: store.Person | None,
) -> None:
    """Log one applied people mutation so it can later be undone. No-op without a thread."""
    write_ledger.record_write(
        _SPEC, settings, thread_id, batch_id, person_id, op, summary,
        json.dumps(asdict(before)) if before else None,
    )


def latest_applied_at(
    settings: Settings, thread_id: str, window_minutes: int
) -> datetime | None:
    """The timestamp of the most recent still-undoable people write on
    ``thread_id`` (or ``None``). Used by the cross-ledger arbiter."""
    return write_ledger.latest_applied_at(_SPEC, settings, thread_id, window_minutes)


def _revert_row(settings: Settings, row: dict) -> str | None:
    """Apply the reverse of one logged people write; return a short summary, or None."""
    try:
        if row["op"] == "add":
            deleted = store.delete_person(settings, row["person_id"])
            return f"removed: {deleted.name}" if deleted else None
        if not row["before_json"]:
            return None
        before = store.Person(**json.loads(row["before_json"]))
        restored = store.restore_person(settings, before)
        return f"restored: {restored.name}"
    except Exception:
        logger.exception("failed to revert person write_log row %s", row["id"])
        return None


def undo_latest(settings: Settings, thread_id: str, window_minutes: int) -> str:
    """Revert the most recent undoable batch of people writes on ``thread_id``."""
    return write_ledger.undo_latest(
        _SPEC, settings, thread_id, window_minutes, _revert_row
    )
