"""Reminder-mute tools — silence nudges without touching calendar or tasks."""
from __future__ import annotations

from ..config import Settings
from ._base import _ISO, _NO_MATCH, ToolContext, ToolSpec, _params


def _resolve_mute_target(settings: Settings, target: str) -> tuple[str, str, str] | None:
    """Resolve ``target`` to ``(scope, target_id, label)``; None when no match
    or ambiguous (the same refuse-don't-guess rule as calendar._target_id)."""
    target = str(target).strip()
    if target.lower() == "all":
        return ("all", "", "all reminders")
    from ..calendar import store as calendar_store
    from ..tasks import store as tasks_store

    events = calendar_store.find_events(settings, target)
    if len(events) == 1:
        return ("event", events[0].id, events[0].title)
    if len(events) > 1:
        return None
    tasks = tasks_store.find_tasks(settings, target)
    if len(tasks) == 1:
        return ("task", tasks[0].id, tasks[0].title)
    return None

def _mute_reminders(ctx: ToolContext, target: str, until: str = "", reason: str = "") -> str:
    from ..calendar.context import format_when, now
    from ..calendar.store import parse_dt
    from ..mutes import set_mute

    resolved = _resolve_mute_target(ctx.settings, target)
    if resolved is None:
        return _NO_MATCH
    scope, target_id, label = resolved
    current = now(ctx.settings)
    if until:
        expiry = parse_dt(str(until))
        if expiry is None:
            return f"Tool failed: until must be {_ISO}."
        if expiry <= current:
            return "Tool failed: until is already in the past."
    else:
        # The ergonomic default: quiet for the rest of today (local time).
        expiry = current.replace(hour=23, minute=59, second=59, microsecond=0)
    set_mute(ctx.settings, scope, target_id, expiry, str(reason), current)
    return f"Muted reminders for {label} until {format_when(ctx.settings, expiry.isoformat())}."

def _unmute_reminders(ctx: ToolContext, target: str) -> str:
    from ..mutes import clear_mute

    resolved = _resolve_mute_target(ctx.settings, target)
    if resolved is None:
        return _NO_MATCH
    scope, target_id, label = resolved
    if clear_mute(ctx.settings, scope, target_id):
        return f"Unmuted reminders for {label}."
    return f"No active mute for {label}."

def _reminder_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "mute_reminders",
            "Silence reminder nudges for one event, one task, or everything "
            '(target="all") until a time, without changing the calendar or '
            "tasks. Use when the user declines a reminder or asks for quiet.",
            _params(
                {
                    "target": ("string", 'Event/task id or title, or "all"'),
                    "until": ("string", f"{_ISO} the mute expires (omit = rest of today)"),
                    "reason": ("string", 'Why, e.g. "user is sick"'),
                },
                ["target"],
            ),
            _mute_reminders,
        ),
        ToolSpec(
            "unmute_reminders",
            "Lift a reminder mute so nudges resume.",
            _params(
                {"target": ("string", 'Event/task id or title, or "all"')},
                ["target"],
            ),
            _unmute_reminders,
        ),
    ]
