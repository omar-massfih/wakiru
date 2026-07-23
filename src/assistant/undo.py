"""Cross-subsystem "undo" arbiter.

The calendar, the to-do list, and the people store each keep their own undo
ledger (:mod:`assistant.calendar.undo`, :mod:`assistant.tasks.undo`,
:mod:`assistant.people.undo`). A single user "undo" should revert whichever of
them wrote most recently on the thread — not guess. This module peeks each
ledger's latest still-undoable timestamp and delegates to the winner's
``undo_latest``. When only one subsystem is enabled (or has anything to revert),
it simply forwards to that one.
"""

from __future__ import annotations

from datetime import datetime

from .calendar import undo as calendar_undo
from .config import Settings, get_settings
from .people import undo as people_undo
from .tasks import undo as tasks_undo


def undo_latest(
    settings: Settings | None, thread_id: str, window_minutes: int
) -> str:
    """Revert the most recent undoable batch on ``thread_id``, across calendar,
    tasks, and people. Reverts the single subsystem whose latest write is newer."""
    settings = settings or get_settings()

    # (priority, ledger): priority breaks an exact-instant tie, preferring the
    # later subsystem (people > tasks > calendar) — arbitrary, as before, since a
    # single turn rarely writes more than one and either revert is correct.
    ledgers = [(0, calendar_undo), (1, tasks_undo), (2, people_undo)]
    candidates: list[tuple[datetime, int, object]] = []
    for priority, ledger in ledgers:
        at = ledger.latest_applied_at(settings, thread_id, window_minutes)
        if at is not None:
            candidates.append((at, priority, ledger))

    if not candidates:
        # No ledger has anything recent; return the calendar's phrasing
        # (it distinguishes "nothing" from "nothing recent enough").
        return calendar_undo.undo_latest(settings, thread_id, window_minutes)

    _, _, winner = max(candidates, key=lambda c: (c[0], c[1]))
    return winner.undo_latest(settings, thread_id, window_minutes)  # type: ignore[attr-defined]
