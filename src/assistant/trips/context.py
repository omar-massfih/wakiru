"""The per-turn trip block — silent except when travel is near or underway."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ..config import Settings
from . import store


def _local_time_note(trip: store.Trip, current: datetime) -> str:
    if not trip.timezone:
        return ""
    try:
        local = current.astimezone(ZoneInfo(trip.timezone))
    except Exception:
        return ""
    return f" Local time in {trip.destination} is {local.strftime('%H:%M (%Z)')}."


def trips_context(settings: Settings) -> str:
    """The active trip, or the next departure while it is imminent — else ``""``.

    Runs on the reply path, so it only reads the local store; destination
    weather stays behind the on-demand ``get_weather`` tool.
    """
    from ..calendar.context import now

    current = now(settings)
    today = current.date()
    active = store.active_trip(settings, today)
    if active is not None:
        started = store.parse_date(active.start)
        ends = store.parse_date(active.end)
        if started is None or ends is None:  # unreachable: active implies both
            return ""
        day = (today - started).days + 1
        total = (ends - started).days + 1
        lines = [
            "## Trip in progress",
            f"{active.name} — in {active.destination} until {active.end} "
            f"(day {day} of {total}, home in {(ends - today).days} day(s))."
            + _local_time_note(active, current),
        ]
        if active.notes:
            lines.append(f"Notes: {active.notes}")
        lines.append(
            "Keep this in mind for scheduling and suggestions; `get_weather` "
            "knows the destination's forecast."
        )
        return "\n".join(lines)
    upcoming = store.next_trip(settings, today)
    if upcoming is None:
        return ""
    departs = store.parse_date(upcoming.start)
    if departs is None:  # unreachable: next_trip implies a parseable start
        return ""
    days_out = (departs - today).days
    if days_out > max(settings.trips_context_days, 0):
        return ""
    lines = [
        "## Upcoming trip",
        f"{upcoming.name} — to {upcoming.destination}, {upcoming.start} to "
        f"{upcoming.end} (departs in {days_out} day(s)).",
    ]
    if upcoming.notes:
        lines.append(f"Notes: {upcoming.notes}")
    hint = "Surface anything that needs doing before departure"
    if settings.enable_lists:
        hint += " — a packing list via `add_to_list` works well"
    lines.append(hint + ".")
    return "\n".join(lines)
