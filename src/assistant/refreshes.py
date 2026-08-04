"""Data refreshes that ride the heartbeat wake — plumbing, not judgment.

The mail snapshot, weather forecast, ICS feed mirror, and CalDAV
reconcile+pull used to run on their own asyncio ticker loops; now every
heartbeat wake starts by running whichever of them is due (:func:`run_due`),
so the heartbeat is the single entry point for all background activity. Each
job is stamped in the heartbeat state KV (``refresh:<name>``) and re-runs once
its interval has passed — a failed attempt re-stamps, so an outage costs one
try per interval, not one per wake (the same behavior the old tickers had).
:func:`next_due_at` feeds the wake scheduler, so a model that backs its next
wake far off cannot starve the refreshes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .calendar import remote as calendar_remote
from .calendar import sync as calendar_sync
from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings

logger = logging.getLogger(__name__)

_WEATHER_LOCATION_STATE = "refresh:weather:location"


def caldav_once(settings: Settings) -> dict:
    """One CalDAV cycle: reconcile queued pushes first, then pull the collection.

    Reconcile before pull so a locally-pending edit lands remotely before the
    pull might otherwise defer to (and then re-import) a now-stale server copy.
    """
    reconciled = calendar_sync.reconcile_caldav(settings) if settings.enable_caldav_write else {}
    pulled = calendar_sync.pull_caldav(settings)
    return {"pull": pulled, "reconcile": reconciled}


def _jobs(settings: Settings, current: datetime | None = None) -> tuple:
    """The refresh jobs: ``(name, interval_minutes, enabled, worker)`` rows."""
    # Deferred: the mail extra may not be installed, and weather pulls urllib
    # plumbing this module's importers shouldn't pay for.
    from . import weather
    from .mail import snapshot as mail_snapshot

    return (
        (
            "mail",
            settings.email_snapshot_minutes,
            settings.enable_email and settings.email_snapshot_minutes > 0,
            lambda: mail_snapshot.maybe_refresh(settings),
        ),
        (
            "weather",
            settings.weather_refresh_minutes,
            weather.effective_location_key(settings, current) is not None
            and settings.weather_refresh_minutes > 0,
            lambda: weather.maybe_refresh(settings, current),
        ),
        (
            "feeds",
            settings.calendar_sync_minutes,
            bool(settings.calendar_ics_urls) and settings.calendar_sync_minutes > 0,
            lambda: calendar_sync.pull_feeds(settings),
        ),
        (
            "caldav",
            settings.caldav_sync_minutes,
            calendar_remote.is_configured(settings) and settings.caldav_sync_minutes > 0,
            lambda: caldav_once(settings),
        ),
    )


def run_due(settings: Settings) -> dict[str, bool]:
    """Run every enabled refresh whose interval has passed; return what ran.

    Each job stamps after the attempt, success or failure alike, so an outage
    is retried on the next interval rather than on every wake. Best-effort:
    one failing job never blocks the others or the wake itself.
    """
    from . import heartbeat

    current = now(settings)
    ran: dict[str, bool] = {}
    for name, interval, enabled, worker in _jobs(settings, current):
        if not enabled:
            continue
        stamp = parse_dt(heartbeat.state_get(settings, f"refresh:{name}"))
        location_key: str | None = None
        location_changed = False
        if name == "weather":
            from . import weather

            location_key = weather.effective_location_key(settings, current)
            attempted = heartbeat.state_get(settings, _WEATHER_LOCATION_STATE)
            location_changed = location_key != attempted
        if (
            not location_changed
            and stamp is not None
            and current - stamp < timedelta(minutes=interval)
        ):
            continue
        try:
            worker()
            ran[name] = True
        except Exception:
            logger.exception("%s refresh failed", name)
            ran[name] = False
        heartbeat.state_set(settings, f"refresh:{name}", current.isoformat(timespec="seconds"))
        if name == "weather" and location_key is not None:
            heartbeat.state_set(settings, _WEATHER_LOCATION_STATE, location_key)
    return ran


def next_due_at(settings: Settings, current: datetime) -> list[datetime]:
    """When each enabled refresh next comes due — the anti-starvation cap for
    the wake scheduler. A job that has never run is due now."""
    from . import heartbeat

    due: list[datetime] = []
    weather_scheduled = False
    for name, interval, enabled, _worker in _jobs(settings, current):
        if not enabled:
            continue
        stamp = parse_dt(heartbeat.state_get(settings, f"refresh:{name}"))
        if name == "weather":
            from . import weather

            weather_scheduled = True
            location_key = weather.effective_location_key(settings, current)
            if location_key != heartbeat.state_get(settings, _WEATHER_LOCATION_STATE):
                due.append(current)
                continue
            cadence_due = current if stamp is None else stamp + timedelta(minutes=interval)
            boundary = weather.next_location_boundary(settings, current)
            due.append(min(cadence_due, boundary) if boundary is not None else cadence_due)
            continue
        due.append(current if stamp is None else stamp + timedelta(minutes=interval))

    # With no configured home location, weather is not an enabled job before a
    # trip starts.  Its start boundary must still wake the scheduler so the trip
    # destination can make weather effective.
    if (
        not weather_scheduled
        and settings.enable_weather
        and settings.weather_refresh_minutes > 0
        and settings.enable_trips
    ):
        from . import weather

        boundary = weather.next_location_boundary(settings, current)
        if boundary is not None:
            due.append(boundary)
    return due
