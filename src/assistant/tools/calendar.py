"""Calendar tools — create/reschedule/cancel/skip/move + free-time search."""
from __future__ import annotations

from ._base import _ISO, _NO_MATCH, ToolContext, ToolSpec, _int_arg, _op_runner, _params


def _calendar_op(ctx: ToolContext, op: dict) -> str:
    from ..calendar import ops as calendar_ops

    result = calendar_ops.apply_op(ctx.settings, op, ctx.thread_id, ctx.batch_id)
    return result or _NO_MATCH

def _find_free_time(
    ctx: ToolContext,
    duration_minutes: str = "60",
    window_start: str = "",
    window_end: str = "",
    earliest_hour: str = "",
    latest_hour: str = "",
) -> str:
    from datetime import timedelta

    from ..calendar import context as calendar_context
    from ..calendar.store import parse_dt

    settings = ctx.settings
    minutes = _int_arg(duration_minutes, 60)
    earliest = _int_arg(earliest_hour, 8)
    latest = _int_arg(latest_hour, 22)
    if minutes is None or minutes <= 0:
        return "duration_minutes must be a positive number of minutes."
    if (
        earliest is None or latest is None
        or not 0 <= earliest < latest <= 24
    ):
        return "earliest_hour/latest_hour must satisfy 0 <= earliest < latest <= 24."
    start = parse_dt(str(window_start)) or calendar_context.now(settings)
    end = parse_dt(str(window_end)) or start + timedelta(days=7)
    if end <= start:
        return "window_end is not after window_start — swap or widen the window."
    slots = calendar_context.free_slots(
        settings, start, end, timedelta(minutes=minutes),
        earliest_hour=earliest, latest_hour=latest,
    )
    if not slots:
        return (
            f"No free slot of {minutes} minutes between "
            f"{calendar_context.format_when(settings, start.isoformat())} and "
            f"{calendar_context.format_when(settings, end.isoformat())} "
            f"(within {earliest:02d}:00-{latest:02d}:00)."
        )
    shown = slots[:8]
    tz = calendar_context.resolve_tz(settings)
    lines = [
        f"- {calendar_context.format_when(settings, a.isoformat())} until "
        f"{b.astimezone(tz).strftime('%H:%M')}"
        f" ({int((b - a).total_seconds() // 60)} min open)"
        for a, b in shown
    ]
    more = f"\n(and {len(slots) - len(shown)} more)" if len(slots) > len(shown) else ""
    return "Free slots:\n" + "\n".join(lines) + more

def _calendar_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "find_free_time",
            "List open calendar gaps — use for 'when am I free?' and picking "
            "meeting slots.",
            _params(
                {
                    "duration_minutes": ("string", "Minimum minutes (default 60)"),
                    "window_start": ("string", f"{_ISO} (default now)"),
                    "window_end": ("string", f"{_ISO} (default a week out)"),
                    "earliest_hour": ("string", "Local hour bound, default 8"),
                    "latest_hour": ("string", "Local hour bound, default 22 (max 24)"),
                },
                [],
            ),
            _find_free_time,
        ),
        ToolSpec(
            "create_event",
            "Schedule a new calendar event.",
            _params(
                {
                    "title": ("string", "Short event title"),
                    "start": ("string", _ISO),
                    "end": ("string", f"{_ISO} (omit for a default 1h)"),
                    "location": ("string", "Where"),
                    "notes": ("string", "Free-form notes"),
                    "rrule": ("string", "RFC 5545 RRULE for a repeating event"),
                },
                ["title", "start"],
            ),
            _op_runner(_calendar_op, "create"),
        ),
        ToolSpec(
            "reschedule_event",
            "Change an existing event's time or details (whole series if recurring).",
            _params(
                {
                    "id": ("string", "Exact event id from Upcoming events"),
                    "start": ("string", _ISO),
                    "end": ("string", _ISO),
                    "title": ("string", "New title"),
                    "location": ("string", "New location"),
                    "notes": ("string", "New notes"),
                },
                ["id"],
            ),
            _op_runner(_calendar_op, "reschedule"),
        ),
        ToolSpec(
            "cancel_event",
            "Cancel an event (whole series if recurring).",
            _params({"id": ("string", "Exact event id")}, ["id"]),
            _op_runner(_calendar_op, "cancel"),
        ),
        ToolSpec(
            "skip_occurrence",
            "Drop a single occurrence of a recurring event — e.g. the user is "
            "skipping it today. Also stops that occurrence's remaining reminder "
            "nudges.",
            _params(
                {
                    "id": ("string", "Series id"),
                    "occurrence": ("string", f"{_ISO} of the occurrence to drop"),
                },
                ["id", "occurrence"],
            ),
            _op_runner(_calendar_op, "skip"),
        ),
        ToolSpec(
            "move_occurrence",
            "Move a single occurrence of a recurring event, leaving the series.",
            _params(
                {
                    "id": ("string", "Series id"),
                    "occurrence": ("string", f"{_ISO} of the original occurrence"),
                    "start": ("string", f"New start, {_ISO}"),
                    "end": ("string", _ISO),
                    "title": ("string", "New title for this occurrence"),
                    "location": ("string", "New location for this occurrence"),
                },
                ["id", "occurrence", "start"],
            ),
            _op_runner(_calendar_op, "move"),
        ),
    ]
