"""The assistant's local calendar (time awareness + scheduling).

A self-contained, offline calendar — no external service, no OAuth. Two paths,
mirroring the memory subsystem:

* **Read** — :func:`agenda_context` injects the current time and upcoming events
  into every turn (wired in :mod:`assistant.agent`), so the model has a clock and
  knows what's scheduled.
* **Write** — the agent's calendar tools (:mod:`assistant.tools`) create,
  reschedule, and cancel events through :func:`.ops.apply_op`.

Events live in a SQLite store (:mod:`.store`) under the memory directory.
"""

from __future__ import annotations

from . import store
from .context import (
    agenda_context,
    busy_events,
    now,
    overlapping_events,
    resolve_tz,
    upcoming_events,
)
from .reminders import due_reminders, run_reminders
from .store import Event
from .undo import undo_latest

__all__ = [
    "Event",
    "agenda_context",
    "busy_events",
    "due_reminders",
    "now",
    "overlapping_events",
    "resolve_tz",
    "run_reminders",
    "store",
    "undo_latest",
    "upcoming_events",
]
