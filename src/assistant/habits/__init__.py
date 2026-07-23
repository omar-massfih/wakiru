"""The health / habits log — a time-series record of what the user does.

A self-contained subsystem: an append-only SQLite log (:mod:`.store`) of habit
entries (a name, an optional value + unit, a date) and the habit tools
(:mod:`assistant.tools`), which log entries and summarize streaks and trends
(:mod:`.context`). Distinct from recurring tasks, which *remind* the user to do
something; this records that they did it, and the numbers.
"""

from __future__ import annotations

from . import context, store
from .store import HabitEntry

__all__ = ["HabitEntry", "context", "store"]
