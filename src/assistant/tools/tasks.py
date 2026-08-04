"""Task tools — add/complete/update/remove over the to-do store."""
from __future__ import annotations

from datetime import timedelta

from ._base import (
    _ISO,
    _NO_MATCH,
    ToolContext,
    ToolSpec,
    _int_arg,
    _op_runner,
    _params,
)


def _task_op(ctx: ToolContext, op: dict) -> str:
    from ..tasks import ops as task_ops

    result = task_ops.apply_op(ctx.settings, op, ctx.thread_id, ctx.batch_id)
    return result or _NO_MATCH


def _task_history(
    ctx: ToolContext, days: object = 7, query: object = "", limit: object = 25
) -> str:
    from ..calendar.context import format_when, now
    from ..tasks import store

    parsed_days = _int_arg(days, 7)
    parsed_limit = _int_arg(limit, 25)
    if parsed_days is None or parsed_days < 1:
        return "Tool failed: days must be a positive integer."
    if parsed_limit is None or parsed_limit < 1:
        return "Tool failed: limit must be a positive integer."
    parsed_days = min(parsed_days, 3650)
    parsed_limit = min(parsed_limit, 100)
    current = now(ctx.settings)
    completions = store.list_task_completions(
        ctx.settings,
        since=current - timedelta(days=parsed_days),
        until=current,
        query=str(query or ""),
        limit=parsed_limit,
    )
    if not completions:
        return "No completed tasks found."
    lines = []
    for item in completions:
        line = f"- {format_when(ctx.settings, item.completed_at)} — {item.title}"
        if item.due:
            line += f" (occurrence due {format_when(ctx.settings, item.due)})"
        if item.rrule:
            line += " (recurring)"
        line += f"  [task id: {item.task_id}]"
        lines.append(line)
    return "Completed tasks:\n" + "\n".join(lines)

def _task_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "task_history",
            "List completed task occurrences.",
            _params(
                {
                    "days": ("integer", "Lookback days; default 7"),
                    "query": ("string", "Title text or task id"),
                    "limit": ("integer", "Result cap; default 25, max 100"),
                },
                [],
            ),
            _task_history,
        ),
        ToolSpec(
            "add_task",
            "Add a to-do, optionally with a due time. Use the calendar for meetings.",
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
                    "notify_only": (
                        "string",
                        "\"true\" for a one-time informational nudge that fires at "
                        "its due time and does NOT keep nagging once overdue — a "
                        "plain \"remind me at TIME that X\", not a to-do to complete",
                    ),
                },
                ["title"],
            ),
            _op_runner(_task_op, "add"),
        ),
        ToolSpec(
            "complete_task",
            "Complete a task; recurring tasks roll to their next due.",
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
                    "notify_only": ("string", "\"true\"/\"false\" one-time nudge flag"),
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
