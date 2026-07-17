"""The calendar read path: the temporal context injected into every turn.

Two things the model otherwise lacks: a clock and a view of what's scheduled.
:func:`agenda_context` renders both into a plain-text block that the agent graph
prepends as a ``SystemMessage`` before the Codex call — the same mechanism recall
uses for memories. It is also handed to the write-path extractor so it can resolve
"tomorrow at noon" against the real current time and reconcile against what's
already booked.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import Settings, get_settings
from . import recurrence, store
from .store import Event

# Assumed duration for an event that has no explicit end, when reasoning about
# overlaps / availability. An hour is a sensible default appointment length.
_DEFAULT_EVENT_MINUTES = 60


def resolve_tz(settings: Settings) -> tzinfo:
    """The timezone the assistant reasons in (configured, else system-local)."""
    if settings.timezone:
        try:
            return ZoneInfo(settings.timezone)
        except ZoneInfoNotFoundError:
            pass
    local = datetime.now().astimezone().tzinfo
    return local or ZoneInfo("UTC")


def now(settings: Settings) -> datetime:
    """Current timezone-aware ``datetime`` in the assistant's timezone."""
    return datetime.now(resolve_tz(settings))


def format_when(settings: Settings, iso: str) -> str:
    """Human-readable local datetime for user-facing summaries."""
    dt = store.parse_dt(iso)
    if dt is None:
        return iso
    return dt.astimezone(resolve_tz(settings)).strftime("%a %d %b %Y %H:%M")


def _render_event(event: Event, settings: Settings, with_id: bool) -> str:
    when = format_when(settings, event.start)
    tz = resolve_tz(settings)
    line = f"- {when} — {event.title}"
    if event.location:
        line += f" @ {event.location}"
    end = store.parse_dt(event.end)
    if end:
        line += f" (until {end.astimezone(tz).strftime('%H:%M')})"
    if event.rrule:  # a series master (occurrences carry no rrule)
        line += f" ({recurrence.humanize_rrule(event.rrule)})"
    if with_id:
        line += f"  [id: {event.id}]"
    return line


def upcoming_events(settings: Settings) -> list[Event]:
    """Occurrences from now through ``calendar_upcoming_days``, capped, soonest first.

    Recurring series are expanded into concrete occurrences within the horizon (see
    :mod:`.recurrence`), so a weekly standup shows up as its next few dates.
    """
    current = now(settings)
    horizon = current + timedelta(days=settings.calendar_upcoming_days)
    events = recurrence.occurrences_in(settings, current, horizon)
    return events[: settings.calendar_max_events]


def writer_view(settings: Settings) -> list[Event]:
    """The schedule shown to the write-path extractor: one-shots + series masters.

    Unlike :func:`upcoming_events`, recurring events appear once as their master row
    (rrule intact) rather than as expanded occurrences, so the extractor sees a
    single line with a stable id to reschedule or cancel the whole series. One-shot
    events are limited to the upcoming horizon; series are always included so an old
    but still-active series stays targetable.
    """
    current = now(settings)
    horizon = current + timedelta(days=settings.calendar_upcoming_days)
    events: list[Event] = []
    for master in store.list_events(settings):
        if master.rrule:
            events.append(master)
        else:
            dt = store.parse_dt(master.start)
            if dt is not None and current <= dt <= horizon:
                events.append(master)
    return sorted(events, key=store._sort_key)[: settings.calendar_max_events]


def render_events(settings: Settings, events: list[Event], with_ids: bool = False) -> str:
    """Render a list of events as text (optionally exposing ids for the writer)."""
    if not events:
        return "(no upcoming events)"
    return "\n".join(_render_event(e, settings, with_ids) for e in events)


def event_interval(settings: Settings, event: Event) -> tuple[datetime, datetime] | None:
    """The ``[start, end)`` instants an event occupies, or ``None`` if its start
    is unparseable. An event with no (or a non-positive) end is treated as lasting
    :data:`_DEFAULT_EVENT_MINUTES`."""
    start = store.parse_dt(event.start)
    if start is None:
        return None
    end = store.parse_dt(event.end)
    if end is None or end <= start:
        end = start + timedelta(minutes=_DEFAULT_EVENT_MINUTES)
    return start, end


def busy_events(settings: Settings, start: datetime, end: datetime) -> list[Event]:
    """Events (expanding recurring series into occurrences) that overlap the
    half-open interval ``[start, end)``, soonest first — the "busy" set for an
    availability question like "am I free Thursday 2-4pm?"."""
    # Pad the scan window so an event starting just before `start` but running
    # into it (or a recurring occurrence near the edge) is still considered.
    pad = timedelta(days=1)
    busy: list[Event] = []
    for event in recurrence.occurrences_in(settings, start - pad, end + pad):
        interval = event_interval(settings, event)
        if interval is None:
            continue
        e_start, e_end = interval
        if e_start < end and start < e_end:  # half-open overlap
            busy.append(event)
    return busy


def free_slots(
    settings: Settings,
    window_start: datetime,
    window_end: datetime,
    duration: timedelta,
    earliest_hour: int = 8,
    latest_hour: int = 22,
) -> list[tuple[datetime, datetime]]:
    """Open gaps of at least ``duration`` within ``[window_start, window_end)``.

    Deterministic complement of :func:`busy_events`: the merged busy intervals
    are subtracted from each day's ``[earliest_hour, latest_hour)`` (assistant's
    timezone), and gaps long enough survive. The past never counts as free —
    the window is clipped to now. Soonest first.
    """
    tz = resolve_tz(settings)
    window_start = max(window_start, now(settings))
    if window_end <= window_start or duration <= timedelta(0):
        return []

    merged: list[tuple[datetime, datetime]] = []
    intervals = sorted(
        i for e in busy_events(settings, window_start, window_end)
        if (i := event_interval(settings, e)) is not None
    )
    for b_start, b_end in intervals:
        if merged and b_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b_end))
        else:
            merged.append((b_start, b_end))

    slots: list[tuple[datetime, datetime]] = []
    # Iterate calendar dates (not +24h steps) so day windows keep their
    # wall-clock hours across a DST change.
    current = window_start.astimezone(tz).date()
    last = window_end.astimezone(tz).date()
    while current <= last:
        cursor = max(datetime.combine(current, time(earliest_hour), tzinfo=tz), window_start)
        close = (
            datetime.combine(current + timedelta(days=1), time(0), tzinfo=tz)
            if latest_hour >= 24  # "until midnight": time(24) doesn't exist
            else datetime.combine(current, time(latest_hour), tzinfo=tz)
        )
        day_close = min(close, window_end)
        for b_start, b_end in merged:
            if b_end <= cursor or b_start >= day_close:
                continue
            if b_start - cursor >= duration:
                slots.append((cursor, b_start))
            cursor = max(cursor, b_end)
        if day_close - cursor >= duration:
            slots.append((cursor, day_close))
        current += timedelta(days=1)
    return slots


def overlapping_events(
    settings: Settings, event: Event, ignore_id: str | None = None
) -> list[Event]:
    """Existing events that overlap ``event``'s time span. ``ignore_id`` excludes
    a given event id (and all its recurring occurrences) — pass the event's own id
    when checking an already-stored event so it doesn't conflict with itself."""
    interval = event_interval(settings, event)
    if interval is None:
        return []
    start, end = interval
    return [e for e in busy_events(settings, start, end) if e.id != ignore_id]


def _part_of_day(hour: int) -> str:
    """A coarse label so the model can greet and pace itself naturally."""
    if 5 <= hour < 9:
        return "early morning"
    if 9 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late night"


def agenda_context(settings: Settings | None = None) -> str:
    """The current-time + upcoming-events block injected ahead of the user's turn."""
    settings = settings or get_settings()
    current = now(settings)
    stamp = current.strftime("%A, %d %B %Y, %H:%M %Z").strip()
    parts = [
        "## Current date and time",
        f"It is currently {stamp} ({_part_of_day(current.hour)}).",
        "",
        "## Upcoming events",
        render_events(settings, upcoming_events(settings)),
    ]
    return "\n".join(parts)
