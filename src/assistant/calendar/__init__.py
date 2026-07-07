"""The assistant's local calendar (time awareness + scheduling).

A self-contained, offline calendar — no external service, no OAuth. Two paths,
mirroring the memory subsystem:

* **Read** — :func:`agenda_context` injects the current time and upcoming events
  into every turn (wired in :mod:`assistant.agent`), so the model has a clock and
  knows what's scheduled.
* **Write** — :func:`update_calendar` runs a reconciling extractor after each turn
  (in the background, off the reply path) to create, reschedule, or cancel events
  from natural language.

Events live in a SQLite store (:mod:`.store`) under the memory directory.
"""

from __future__ import annotations

from . import store
from .context import agenda_context, now, resolve_tz, upcoming_events
from .ops import update_calendar
from .reminders import due_reminders, run_reminders
from .store import Event
from .undo import undo_latest

__all__ = [
    "Event",
    "store",
    "agenda_context",
    "now",
    "resolve_tz",
    "upcoming_events",
    "update_calendar",
    "due_reminders",
    "run_reminders",
    "undo_latest",
]
