"""Cross-subsystem "undo" arbiter.

The calendar and the to-do list each keep their own undo ledger
(:mod:`assistant.calendar.undo`, :mod:`assistant.tasks.undo`). A single user
"undo" should revert whichever of them wrote most recently on the thread — not
guess. This module peeks both ledgers' latest still-undoable timestamp and
delegates to the winner's ``undo_latest``. When only one subsystem is enabled
(or has anything to revert), it simply forwards to that one.
"""

from __future__ import annotations

from .calendar import undo as calendar_undo
from .config import Settings, get_settings
from .tasks import undo as tasks_undo


def undo_latest(
    settings: Settings | None, thread_id: str, window_minutes: int
) -> str:
    """Revert the most recent undoable batch on ``thread_id``, across calendar
    and tasks. Reverts the single subsystem whose latest write is newer."""
    settings = settings or get_settings()

    cal_at = calendar_undo.latest_applied_at(settings, thread_id, window_minutes)
    task_at = tasks_undo.latest_applied_at(settings, thread_id, window_minutes)

    if cal_at is None and task_at is None:
        # Neither ledger has anything recent; return the calendar's phrasing
        # (it distinguishes "nothing" from "nothing recent enough").
        return calendar_undo.undo_latest(settings, thread_id, window_minutes)

    # Whichever wrote more recently owns this undo. Ties (same instant) favor
    # tasks arbitrarily — a single turn rarely writes both, and either revert is
    # a correct "undo the last thing".
    if task_at is not None and (cal_at is None or task_at >= cal_at):
        return tasks_undo.undo_latest(settings, thread_id, window_minutes)
    return calendar_undo.undo_latest(settings, thread_id, window_minutes)
