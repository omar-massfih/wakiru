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

    Runs on the reply path, so it only reads local stores. The cached weather
    block follows an active destination when weather is enabled.
    """
    if not settings.enable_trips:
        return ""

    from ..calendar.context import now, resolve_home_tz

    current = now(settings)
    active = store.active_trip_at(settings, current, resolve_home_tz(settings))
    if active is not None:
        try:
            active_tz = ZoneInfo(active.timezone) if active.timezone else resolve_home_tz(settings)
        except Exception:
            active_tz = resolve_home_tz(settings)
        today = current.astimezone(active_tz).date()
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
        guidance = "Keep this in mind for scheduling and suggestions."
        if settings.enable_weather:
            from .. import weather

            if weather.current(settings, current):
                guidance += " Cached weather follows this destination."
            guidance += " Use `get_weather` for places or days not covered by cached weather."
        lines.append(guidance)
        return "\n".join(lines)
    home_tz = resolve_home_tz(settings)
    upcoming = store.next_trip_at(settings, current, home_tz)
    if upcoming is None:
        return ""
    departs = store.parse_date(upcoming.start)
    if departs is None:  # unreachable: next_trip implies a parseable start
        return ""
    try:
        upcoming_tz = ZoneInfo(upcoming.timezone) if upcoming.timezone else home_tz
    except Exception:
        upcoming_tz = home_tz
    days_out = (departs - current.astimezone(upcoming_tz).date()).days
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
