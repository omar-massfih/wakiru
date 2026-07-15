"""Undo ledger — the calendar's confirmation/safety-net path.

Every applied write (create/reschedule/cancel/skip/move) is logged, grouped
per turn by a ``batch_id`` (see :mod:`.ops`), so replying "undo" reverts
exactly what one turn changed — deterministically, with no LLM call involved.
The generic driver lives in :mod:`assistant.write_ledger`; this module
supplies the calendar's spec and the event-specific revert.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime

from .. import write_ledger
from ..config import Settings
from . import store
from .context import format_when

logger = logging.getLogger(__name__)

_SPEC = write_ledger.LedgerSpec(
    kind="calendar",
    db_path=lambda settings: settings.calendar_db_path,
    target_column="event_id",
    pg_record="record_calendar_write",
    pg_rows="calendar_write_rows",
    pg_mark_undone="mark_calendar_writes_undone",
)


def _connect(settings: Settings):
    """One transaction on this ledger's local DB (see write_ledger.connect)."""
    return write_ledger.connect(_SPEC, settings)


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
    write_ledger.record_write(
        _SPEC, settings, thread_id, batch_id, event_id, op, summary,
        json.dumps(asdict(before)) if before else None,
    )


def latest_applied_at(
    settings: Settings, thread_id: str, window_minutes: int
) -> datetime | None:
    """Timestamp of the most recent still-undoable write on ``thread_id`` (or
    ``None`` if nothing is recent enough). Lets the cross-ledger arbiter in
    :mod:`assistant.undo` decide whether the calendar or the tasks ledger owns
    the most recent write."""
    return write_ledger.latest_applied_at(_SPEC, settings, thread_id, window_minutes)


def _revert_row(settings: Settings, row: dict) -> str | None:
    """Apply the reverse of one logged write; return a short summary, or None on failure.

    The local revert is mirrored to CalDAV best-effort: undoing a *create* issues a
    remote **DELETE** (the roadmap's "undo mapped to remote deletes"); undoing any
    other write re-PUTs the restored event. The remote step never fails the undo — a
    push that can't land is queued to the outbox for reconcile.
    """
    # Lazy import: ops imports undo, so importing ops at module load would cycle.
    from .ops import _push_caldav

    try:
        if row["op"] == "create":
            deleted = store.delete_event(settings, row["event_id"])
            if deleted is None:
                return None
            _push_caldav(settings, None, "cancel", deleted)
            return f"removed: {deleted.title}"
        if not row["before_json"]:
            return None
        before = store.Event(**json.loads(row["before_json"]))
        restored = store.restore_event(settings, before)
        _push_caldav(settings, restored, "reschedule", None)
        return f"restored: {restored.title} @ {format_when(settings, restored.start)}"
    except Exception:
        logger.exception("failed to revert write_log row %s", row["id"])
        return None


def undo_latest(settings: Settings, thread_id: str, window_minutes: int) -> str:
    """Revert the most recent undoable batch of calendar writes on ``thread_id``."""
    return write_ledger.undo_latest(
        _SPEC, settings, thread_id, window_minutes, _revert_row
    )
