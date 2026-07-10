"""The tasks read path: the open to-do list injected into every turn.

:func:`tasks_context` renders the open tasks (flagging any that are overdue
against the assistant's clock) into a plain-text block that the agent graph
prepends as a ``SystemMessage`` before the model call — the same mechanism recall
and the calendar agenda use. It is also handed to the write-path extractor so it
can target existing tasks by id.
"""

from __future__ import annotations

from ..calendar.context import format_when, now
from ..calendar.store import parse_dt
from ..config import Settings, get_settings
from . import store
from .store import Task


def _render_task(task: Task, settings: Settings, with_id: bool, current) -> str:
    line = f"- {task.title}"
    due = parse_dt(task.due)
    if due is not None:
        line += f" (due {format_when(settings, task.due)}"
        line += ", OVERDUE)" if due < current else ")"
    if with_id:
        line += f"  [id: {task.id}]"
    return line


def render_tasks(settings: Settings, tasks: list[Task], with_ids: bool = False) -> str:
    """Render open tasks as text (optionally exposing ids for the writer)."""
    if not tasks:
        return "(no open tasks)"
    current = now(settings)
    return "\n".join(_render_task(t, settings, with_ids, current) for t in tasks)


def open_tasks(settings: Settings) -> list[Task]:
    """Open tasks, soonest-due first, capped by ``tasks_max_open``."""
    return store.list_tasks(settings, include_done=False)[: settings.tasks_max_open]


def tasks_context(settings: Settings | None = None) -> str:
    """The open-tasks block injected ahead of the user's turn."""
    settings = settings or get_settings()
    return "## Open tasks\n" + render_tasks(settings, open_tasks(settings))
