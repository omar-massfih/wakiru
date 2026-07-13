"""The assistant's to-do list (tasks with a done state and an optional due date).

A self-contained, offline subsystem mirroring :mod:`assistant.calendar`, but for
work that has no fixed meeting time:

* **Read** — :func:`tasks_context` injects the open to-do list into every turn
  (wired in :mod:`assistant.agent`), so the model knows what's outstanding.
* **Write** — the agent's task tools (:mod:`assistant.tools`) add, complete,
  update, and remove tasks through :func:`.ops.apply_op`.

Tasks live in a SQLite store (:mod:`.store`) under the memory directory, with a
parallel undo ledger (:mod:`.undo`) that the cross-subsystem "undo" arbiter
(:mod:`assistant.undo`) consults alongside the calendar's.
"""

from __future__ import annotations

from . import store
from .context import open_tasks, render_tasks, tasks_context
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
]
