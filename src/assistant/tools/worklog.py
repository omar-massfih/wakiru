"""Work-log tools — a start/stop timer and direct logs, rolled up per project."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _int_arg, _params


def _start_work(ctx: ToolContext, **args: object) -> str:
    from ..worklog import store
    from ..worklog.context import fmt_minutes

    project = str(args.get("project", "")).strip()
    if not project:
        return "Tool failed: a project name is required."
    started, stopped = store.start_entry(
        ctx.settings, project, note=str(args.get("note", "") or "")
    )
    text = f"Clock started on {started.project}."
    if stopped is not None:
        text += f" (Stopped {stopped.project} first — {fmt_minutes(stopped.minutes)}.)"
    return text


def _stop_work(ctx: ToolContext, **args: object) -> str:
    from ..worklog import store
    from ..worklog.context import fmt_minutes

    entry = store.stop_entry(ctx.settings, note=str(args.get("note", "") or ""))
    if entry is None:
        return "No work timer is running."
    return f"Stopped: {entry.project} — {fmt_minutes(entry.minutes)} logged ({entry.worked_on})."


def _log_work(ctx: ToolContext, **args: object) -> str:
    from ..worklog import store
    from ..worklog.context import fmt_minutes

    entry = store.log_entry(
        ctx.settings,
        project=str(args.get("project", "") or ""),
        minutes=args.get("minutes", 0),
        note=str(args.get("note", "") or ""),
        on=str(args.get("on", "") or ""),
    )
    if entry is None:
        return "Tool failed: a project and a positive number of minutes are required."
    return f"Logged {fmt_minutes(entry.minutes)} on {entry.project} ({entry.worked_on})."


def _work_summary(ctx: ToolContext, **args: object) -> str:
    from ..worklog import store
    from ..worklog.context import summary

    days = _int_arg(args.get("days", ""), 7)
    if days is None or days <= 0:
        days = 7
    return summary(ctx.settings, store._today(ctx.settings), days=days)


def _remove_work_entry(ctx: ToolContext, **args: object) -> str:
    from ..worklog import store
    from ..worklog.context import fmt_minutes

    entry_id = str(args.get("id", "")).strip()
    if not entry_id:
        return "Tool failed: an entry id is required (see work_summary)."
    removed = store.delete_entry(ctx.settings, entry_id)
    if removed is None:
        return f"No work entry with id {entry_id!r}."
    return (
        f"Removed the {fmt_minutes(removed.minutes)} {removed.project} entry"
        f" from {removed.worked_on}."
    )


def _worklog_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "start_work",
            "Start the work clock on a project when the user says they are "
            "starting on something (\"starting on the report\", \"back to "
            "client X\"). Any already-running timer is stopped and logged "
            "first, so switching tasks is one call.",
            _params(
                {
                    "project": ("string", "What they are working on, e.g. \"wakiru\", \"client X\""),
                    "note": ("string", "Optional detail, e.g. \"code review\""),
                },
                ["project"],
            ),
            _start_work,
        ),
        ToolSpec(
            "stop_work",
            "Stop the running work timer and log its duration — when the user "
            "says they are done, taking a break, or heading out.",
            _params({"note": ("string", "Optional note about what got done")}, []),
            _stop_work,
        ),
        ToolSpec(
            "log_work",
            "Record a finished stretch of work after the fact — \"2 hours on "
            "the budget yesterday\". Convert what they said to whole minutes.",
            _params(
                {
                    "project": ("string", "What they worked on"),
                    "minutes": ("string", "Duration in minutes, e.g. \"120\""),
                    "on": ("string", "The date as YYYY-MM-DD; omit for today"),
                    "note": ("string", "Optional detail"),
                },
                ["project", "minutes"],
            ),
            _log_work,
        ),
        ToolSpec(
            "work_summary",
            "Roll up logged work time — today and the last N days per project, "
            "plus recent entries with ids (\"how much did I work this week?\").",
            _params({"days": ("string", "Window in days; omit for 7")}, []),
            _work_summary,
        ),
        ToolSpec(
            "remove_work_entry",
            "Delete a single work-log entry by its id (from work_summary) — "
            "for correcting a mistaken log or an accidental timer.",
            _params({"id": ("string", "Exact entry id")}, ["id"]),
            _remove_work_entry,
        ),
    ]
