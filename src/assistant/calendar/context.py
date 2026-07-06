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
from . import store
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
    if with_id:
        line += f"  [id: {event.id}]"
    return line


def upcoming_events(settings: Settings) -> list[Event]:
    """Events from now through ``calendar_upcoming_days``, capped, soonest first."""
    current = now(settings)
    horizon = current + timedelta(days=settings.calendar_upcoming_days)
    events = store.list_events(settings, start_from=current, start_to=horizon)
    return events[: settings.calendar_max_events]


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
