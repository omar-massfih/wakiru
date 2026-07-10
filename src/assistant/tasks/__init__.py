"""The assistant's to-do list (tasks with a done state and an optional due date).

A self-contained, offline subsystem mirroring :mod:`assistant.calendar`, but for
work that has no fixed meeting time:

* **Read** — :func:`tasks_context` injects the open to-do list into every turn
  (wired in :mod:`assistant.agent`), so the model knows what's outstanding.
* **Write** — :func:`update_tasks` runs a reconciling extractor after each turn
  (in the background, off the reply path) to add, complete, update, or remove
  tasks from natural language.

Tasks live in a SQLite store (:mod:`.store`) under the memory directory, with a
parallel undo ledger (:mod:`.undo`) that the cross-subsystem "undo" arbiter
(:mod:`assistant.undo`) consults alongside the calendar's.
"""

from __future__ import annotations

from . import store
from .context import open_tasks, render_tasks, tasks_context
from .ops import update_tasks
from .reminders import due_task_reminders, run_task_reminders
from .store import Task

__all__ = [
    "Task",
    "due_task_reminders",
    "open_tasks",
    "render_tasks",
    "run_task_reminders",
    "store",
    "tasks_context",
    "update_tasks",
]
