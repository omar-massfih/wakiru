"""Task tools — add/complete/update/remove over the to-do store."""
from __future__ import annotations

from ._base import _ISO, _NO_MATCH, ToolContext, ToolSpec, _op_runner, _params


def _task_op(ctx: ToolContext, op: dict) -> str:
    from ..tasks import ops as task_ops

    result = task_ops.apply_op(ctx.settings, op, ctx.thread_id, ctx.batch_id)
    return result or _NO_MATCH

def _task_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "add_task",
            "Add a to-do, optionally with a due time. Not a meeting with other "
            "people (use the calendar for those) — but a plain \"remind me at "
            "TIME that X\" IS this: add it with that due time, called "
            "immediately, not schedule_followup.",
            _params(
                {
                    "title": ("string", "Short task title"),
                    "due": ("string", f"Optional due date, {_ISO}"),
                    "notes": ("string", "Free-form notes"),
                    "rrule": (
                        "string",
                        "RFC 5545 RRULE for a recurring chore (needs a due date"
                        " to anchor); completing rolls the due forward",
                    ),
                },
                ["title"],
            ),
            _op_runner(_task_op, "add"),
        ),
        ToolSpec(
            "complete_task",
            "Mark a task done (also stops its reminder nagging). A recurring "
            "task rolls to its next due instead of closing.",
            _params({"id": ("string", "Exact task id from Open tasks")}, ["id"]),
            _op_runner(_task_op, "complete"),
        ),
        ToolSpec(
            "update_task",
            "Change a task's title, due date, notes, or recurrence.",
            _params(
                {
                    "id": ("string", "Exact task id"),
                    "title": ("string", "New title"),
                    "due": ("string", f"New due date, {_ISO}"),
                    "notes": ("string", "New notes"),
                    "rrule": ("string", "New RFC 5545 RRULE"),
                },
                ["id"],
            ),
            _op_runner(_task_op, "update"),
        ),
        ToolSpec(
            "remove_task",
            "Delete a task without completing it.",
            _params({"id": ("string", "Exact task id")}, ["id"]),
            _op_runner(_task_op, "remove"),
        ),
    ]
