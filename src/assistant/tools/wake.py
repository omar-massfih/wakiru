"""Self-pacing tools — the background wake schedules its own next wake."""
from __future__ import annotations

from ._base import _ISO, ToolContext, ToolSpec, _params


def _set_next_wake(ctx: ToolContext, when: str, reason: str = "") -> str:
    from datetime import timedelta

    from .. import heartbeat
    from ..calendar.context import format_when, now
    from ..calendar.store import parse_dt

    settings = ctx.settings
    target = parse_dt(str(when))
    if target is None:
        return f"Tool failed: when must be {_ISO}."
    current = now(settings)
    if target <= current:
        return "Tool failed: when is already in the past."

    # Clamp into the same window next_wake_at enforces, anchored on this wake, so
    # the reported time is the one that will actually fire.
    anchor_raw = heartbeat.state_get(settings, "last_wake_at")
    anchor = parse_dt(anchor_raw) if anchor_raw else None
    anchor = anchor or current
    floor = anchor + timedelta(minutes=max(settings.heartbeat_wake_min_minutes, 0))
    ceiling = anchor + timedelta(
        minutes=settings.heartbeat_wake_max_minutes or max(settings.heartbeat_minutes, 1)
    )
    clamped = min(max(target, floor), ceiling)
    heartbeat.state_set(settings, "next_wake_at", clamped.isoformat(timespec="seconds"))
    heartbeat.state_set(settings, "next_wake_reason", str(reason).strip())
    note = " (clamped to your pacing bounds)" if clamped != target else ""
    return f"Next wake set for {format_when(settings, clamped.isoformat())}{note}."

def _wake_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "set_next_wake",
            "Set when you next wake yourself. Use it to wake right before "
            "something time-sensitive (a meeting, a promised check-in) or to "
            "back off when nothing is happening. The time is clamped to your "
            "pacing bounds; to guarantee delivery of a specific check-in, "
            "schedule a follow-up instead — a self-set wake is still subject to "
            "the ambient push throttle.",
            _params(
                {
                    "when": ("string", _ISO),
                    "reason": ("string", "One line on why, shown to you on that wake"),
                },
                ["when"],
            ),
            _set_next_wake,
        )
    ]
