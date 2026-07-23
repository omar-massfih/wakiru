"""Habit tools — log entries and summarize streaks/trends over the habit log."""
from __future__ import annotations

from ._base import _ISO, ToolContext, ToolSpec, _params


def _log_habit(ctx: ToolContext, **args: object) -> str:
    from ..habits import store

    habit = str(args.get("habit", "")).strip()
    if not habit:
        return "Tool failed: a habit name is required."
    entry = store.log_entry(
        ctx.settings,
        habit=habit,
        value=args.get("value", 0) or 0,
        unit=str(args.get("unit", "") or ""),
        note=str(args.get("note", "") or ""),
        on=str(args.get("on", "") or ""),
    )
    detail = ""
    if entry.value:
        num = int(entry.value) if entry.value == int(entry.value) else round(entry.value, 2)
        detail = f" ({num} {entry.unit})".replace(" )", ")")
    return f"Logged {entry.habit}{detail} on {entry.logged_on}"


def _habit_summary(ctx: ToolContext, **args: object) -> str:
    from datetime import date

    from ..habits import context

    habit = str(args.get("habit", "")).strip()
    today = date.today()
    if habit:
        return context.summarize(ctx.settings, habit, today)
    return context.overview(ctx.settings, today)


def _remove_habit_entry(ctx: ToolContext, **args: object) -> str:
    from ..habits import store

    entry_id = str(args.get("id", "")).strip()
    if not entry_id:
        return "Tool failed: an entry id is required (see habit_summary)."
    removed = store.delete_entry(ctx.settings, entry_id)
    if removed is None:
        return f"No habit entry with id {entry_id!r}."
    return f"Removed the {removed.habit} entry from {removed.logged_on}."


def _habit_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "log_habit",
            "Record that the user did a habit or hit a health metric — a workout, "
            "sleep, weight, steps, a glass of water. Capture the number and unit "
            "when they give one; use it whenever they report doing something "
            "trackable.",
            _params(
                {
                    "habit": ("string", "What it is, e.g. \"gym\", \"sleep\", \"weight\""),
                    "value": ("string", "The number, e.g. \"7.5\" (omit for a plain check-in)"),
                    "unit": ("string", "The unit, e.g. \"hours\", \"kg\", \"km\""),
                    "note": ("string", "Anything extra about this entry"),
                    "on": ("string", f"When, {_ISO} or YYYY-MM-DD; omit for today"),
                },
                ["habit"],
            ),
            _log_habit,
        ),
        ToolSpec(
            "habit_summary",
            "Summarize the user's habit log — streaks, last entry, and recent "
            "trend. Pass a habit name for its detail (with recent entries + ids); "
            "omit it for an overview of everything tracked.",
            _params({"habit": ("string", "A habit name, or omit for the overview")}, []),
            _habit_summary,
        ),
        ToolSpec(
            "remove_habit_entry",
            "Delete a single logged habit entry by its id (from habit_summary) — "
            "for correcting a mistaken log.",
            _params({"id": ("string", "Exact entry id")}, ["id"]),
            _remove_habit_entry,
        ),
    ]
