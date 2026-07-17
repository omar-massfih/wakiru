"""Recurrence: expand a series master into concrete occurrences on read.

A recurring event is stored as a single *master* row (:class:`Event`) carrying an
RFC 5545 rule in ``rrule`` (e.g. ``FREQ=WEEKLY;BYDAY=MO``), with its ``start`` as the
DTSTART. Occurrences are never materialized — they are computed within a query
window here, which is why a series whose DTSTART has slid into the past still keeps
producing future occurrences (unlike the ``start_from=now`` filter in the store).

Each occurrence is a copy of the master with its own ``start``/``end`` and its
``rrule`` cleared, so downstream code (rendering, reminders) treats it as an
ordinary dated event. Occurrences keep the master ``id``: the reminder ledger keys
on ``(event_id, event_start, lead)`` and edits resolve to the series as a whole.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, tzinfo

from dateutil.rrule import rrulestr

from ..config import Settings
from . import store
from .store import Event, parse_dt

logger = logging.getLogger(__name__)

_FREQ_ADVERB = {
    "DAILY": "daily",
    "WEEKLY": "weekly",
    "MONTHLY": "monthly",
    "YEARLY": "yearly",
}
_FREQ_NOUN = {"DAILY": "day", "WEEKLY": "week", "MONTHLY": "month", "YEARLY": "year"}
_WEEKDAY_NAME = {
    "MO": "Monday", "TU": "Tuesday", "WE": "Wednesday", "TH": "Thursday",
    "FR": "Friday", "SA": "Saturday", "SU": "Sunday",
}


def _parse_parts(rule: str) -> dict[str, str]:
    """Split an RRULE string into its uppercase KEY=VALUE parts."""
    parts: dict[str, str] = {}
    for chunk in rule.replace("RRULE:", "").split(";"):
        key, sep, value = chunk.partition("=")
        if sep:
            parts[key.strip().upper()] = value.strip()
    return parts


def validate_rrule(rule: str, dtstart: datetime | None = None) -> bool:
    """True if ``rule`` is a parseable RFC 5545 recurrence rule."""
    if not rule.strip():
        return False
    try:
        rrulestr(rule, dtstart=dtstart or datetime(2000, 1, 1))
        return True
    except (ValueError, TypeError):
        return False


def humanize_rrule(rule: str) -> str:
    """A short human label for a rule ('every Monday', 'daily'); rule itself on doubt."""
    parts = _parse_parts(rule)
    freq = parts.get("FREQ", "")
    try:
        interval = int(parts.get("INTERVAL", "1") or "1")
    except ValueError:
        interval = 1
    byday = parts.get("BYDAY", "")

    if freq == "WEEKLY" and byday and interval == 1:
        days = [_WEEKDAY_NAME.get(d, d) for d in byday.split(",") if d]
        if days:
            return "every " + ", ".join(days)
    if freq in _FREQ_ADVERB:
        if interval == 1:
            return _FREQ_ADVERB[freq]
        return f"every {interval} {_FREQ_NOUN[freq]}s"
    return rule


def build_rule(rule: str, dtstart: datetime, tz: tzinfo | None = None):
    """The dateutil rule for ``rule`` anchored at ``dtstart``, or ``None`` if
    unusable. Shared by calendar events and recurring tasks.

    ``tz`` should be the assistant's timezone (a ZoneInfo when configured).
    Stored stamps round-trip through ISO strings, whose parsed tzinfo is a
    *fixed* UTC offset — expanding a rule from that would pin every occurrence
    to the offset the anchor was created under, drifting an hour across a DST
    change. Converting DTSTART to the zone first keeps occurrences on the same
    wall-clock time year-round.
    """
    if tz is not None:
        dtstart = dtstart.astimezone(tz)
    try:
        return rrulestr(rule, dtstart=dtstart)
    except (ValueError, TypeError):
        return None


def _rule_for(event: Event, tz: tzinfo | None = None):
    """The dateutil rule for a series master, or ``None`` if start/rule is unusable."""
    dtstart = parse_dt(event.start)
    if dtstart is None:
        return None
    rule = build_rule(event.rrule, dtstart, tz)
    if rule is None:
        logger.warning("skipping event %s with invalid rrule %r", event.id, event.rrule)
    return rule


def resolve_occurrence(
    event: Event, when: datetime, tz: tzinfo | None = None
) -> datetime | None:
    """Snap ``when`` to the series' actual occurrence datetime near it.

    Matches an exact instant first, then any occurrence on the same calendar date
    (so "this Monday" resolves even if the LLM's time-of-day is loose). ``None`` if
    the event is not a series or no occurrence lands within a day of ``when``.
    """
    if not event.rrule:
        return None
    rule = _rule_for(event, tz)
    if rule is None:
        return None
    window = rule.between(when - timedelta(days=1), when + timedelta(days=1), inc=True)
    for occ in window:
        if occ == when:
            return occ
    for occ in window:
        if occ.date() == when.date():
            return occ
    return None


def expand(
    event: Event,
    window_start: datetime,
    window_end: datetime,
    tz: tzinfo | None = None,
) -> list[Event]:
    """Occurrences of ``event`` within ``[window_start, window_end]`` (inclusive).

    A non-recurring event yields itself. A recurring master yields one synthetic
    :class:`Event` per occurrence — same id, ``rrule`` cleared, ``end`` shifted by
    the master's original duration. Skipped occurrences (``exdates``) are dropped and
    single-occurrence ``overrides`` are applied in place; a moved occurrence keeps its
    slot in the window even if its new time falls slightly outside it. An unparseable
    master yields nothing (logged).
    """
    if not event.rrule:
        return [event]
    dtstart = parse_dt(event.start)
    rule = _rule_for(event, tz)
    if dtstart is None or rule is None:
        return []

    end_dt = parse_dt(event.end)
    duration = end_dt - dtstart if end_dt is not None else None
    skipped = {parse_dt(x) for x in store.load_exdates(event)}
    overrides = {parse_dt(k): v for k, v in store.load_overrides(event).items()}

    occurrences: list[Event] = []
    for occ in rule.between(window_start, window_end, inc=True):
        if occ in skipped:
            continue
        occ_end = (occ + duration).isoformat() if duration is not None else ""
        item = replace(
            event, start=occ.isoformat(), end=occ_end, rrule="", exdates="", overrides=""
        )
        change = overrides.get(occ)
        if change:
            item = replace(
                item,
                start=change.get("start") or item.start,
                end=change.get("end") or item.end,
                title=change.get("title") or item.title,
                location=change.get("location") or item.location,
                notes=change.get("notes") or item.notes,
            )
        occurrences.append(item)
    return occurrences


def occurrences_in(
    settings: Settings, start_from: datetime, start_to: datetime
) -> list[Event]:
    """All events occurring within ``[start_from, start_to]``, soonest first.

    Reads every master unbounded (so a past-DTSTART series is still considered),
    expands recurring ones into the window, and window-filters one-shot events.
    """
    # Lazy import: context imports this module at top level.
    from .context import resolve_tz

    tz = resolve_tz(settings)
    result: list[Event] = []
    for master in store.list_events(settings):
        if master.rrule:
            result.extend(expand(master, start_from, start_to, tz))
        else:
            dt = parse_dt(master.start)
            if dt is not None and start_from <= dt <= start_to:
                result.append(master)
    return sorted(result, key=store._sort_key)
