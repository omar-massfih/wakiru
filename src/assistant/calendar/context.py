"""The calendar read path: the temporal context injected into every turn.

Two things the model otherwise lacks: a clock and a view of what's scheduled.
:func:`agenda_context` renders both into a plain-text block that the agent graph
prepends as a ``SystemMessage`` before the Codex call — the same mechanism recall
uses for memories. It is also handed to the write-path extractor so it can resolve
"tomorrow at noon" against the real current time and reconcile against what's
already booked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import Settings, get_settings
from . import recurrence, store
from .store import Event


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


def _render_event(event: Event, tz: tzinfo, with_id: bool) -> str:
    dt = store.parse_dt(event.start)
    when = dt.astimezone(tz).strftime("%a %d %b %Y %H:%M") if dt else event.start
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
    tz = resolve_tz(settings)
    if not events:
        return "(no upcoming events)"
    return "\n".join(_render_event(e, tz, with_ids) for e in events)


def agenda_context(settings: Settings | None = None) -> str:
    """The current-time + upcoming-events block injected ahead of the user's turn."""
    settings = settings or get_settings()
    current = now(settings)
    stamp = current.strftime("%A, %d %B %Y, %H:%M %Z").strip()
    parts = [
        "## Current date and time",
        f"It is currently {stamp}.",
        "",
        "## Upcoming events",
        render_events(settings, upcoming_events(settings)),
    ]
    return "\n".join(parts)
