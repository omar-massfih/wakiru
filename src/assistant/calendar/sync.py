"""One-way pull sync from external calendars (ICS subscription URLs).

Google Calendar, Outlook, and every CalDAV provider export a *secret iCal
address*; ``CALENDAR_ICS_URLS`` points at one or more of them and
:func:`pull_feeds` mirrors their events into the local store on the reminder
ticker's cadence. No OAuth, no two-way writes — the feed is the source of
truth, so synced events are **read-only** here (the write path refuses to
reschedule or cancel them; change them in the source calendar instead).

Synced rows are ordinary events with a *stable, feed-derived id*
(``ics`` + feed hash + UID hash — see :func:`_event_id`), so:

* re-pulls upsert in place (via :func:`store.restore_event`) instead of
  duplicating, and deletions in the feed delete the mirrored row;
* the id prefix itself marks the event as synced (:func:`is_synced_id`) — no
  schema change, so the Postgres backend works unchanged;
* agenda context, conflict checks, reminders, and the daily briefing all pick
  synced events up automatically, because they read the same store.

Recurring events keep their RRULE (the store expands occurrences on read);
EXDATEs are mirrored, and RECURRENCE-ID overrides become per-occurrence
overrides on the master.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from datetime import date, datetime

from ..config import Settings, get_settings
from . import store
from .context import resolve_tz

logger = logging.getLogger(__name__)

_SYNCED_PREFIX = "ics"
_FETCH_TIMEOUT_SECONDS = 30
_MAX_FEED_BYTES = 10_000_000

# Fields that participate in change detection between pulls.
_MIRRORED_FIELDS = ("title", "start", "end", "location", "notes", "rrule", "exdates", "overrides")


def is_synced_id(event_id: str) -> bool:
    """Whether an event id belongs to a mirrored (read-only) external event."""
    return event_id.startswith(_SYNCED_PREFIX)


def _feed_prefix(url: str) -> str:
    return _SYNCED_PREFIX + hashlib.sha1(url.encode()).hexdigest()[:3]


def _event_id(url: str, uid: str) -> str:
    return _feed_prefix(url) + hashlib.sha1(uid.encode()).hexdigest()[:6]


def _iso(value: object, settings: Settings) -> str:
    """An ICS DTSTART/DTEND value as a tz-aware ISO string ('' if absent).

    Date-only values (all-day events) become local midnight; floating (naive)
    datetimes are pinned to the assistant's timezone.
    """
    if value is None:
        return ""
    dt = getattr(value, "dt", value)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=resolve_tz(settings))
        return dt.isoformat(timespec="seconds")
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day, tzinfo=resolve_tz(settings)).isoformat()
    return ""


def _text(component, key: str) -> str:
    value = component.get(key)
    return str(value).strip() if value is not None else ""


def _rrule(component) -> str:
    rule = component.get("rrule")
    if rule is None:
        return ""
    try:
        return rule.to_ical().decode().strip()
    except Exception:
        return ""


def _exdates(component, settings: Settings) -> list[str]:
    raw = component.get("exdate")
    if raw is None:
        return []
    out: list[str] = []
    for chunk in raw if isinstance(raw, list) else [raw]:
        for entry in getattr(chunk, "dts", []):
            stamp = _iso(entry, settings)
            if stamp:
                out.append(stamp)
    return out


def parse_feed(text: str, url: str, settings: Settings) -> dict[str, store.Event]:
    """Parse ICS text into upsert-ready events keyed by their feed-derived id."""
    import json

    from icalendar import Calendar

    calendar = Calendar.from_ical(text)
    events: dict[str, store.Event] = {}
    overrides: list[tuple[str, str, dict[str, str]]] = []  # (master id, occurrence, fields)

    for component in calendar.walk("VEVENT"):
        uid = _text(component, "uid")
        start = _iso(component.get("dtstart"), settings)
        if not uid or not start:
            continue
        fields = {
            "title": _text(component, "summary") or "(untitled)",
            "start": start,
            "end": _iso(component.get("dtend"), settings),
            "location": _text(component, "location"),
            "notes": _text(component, "description"),
        }
        recurrence_id = _iso(component.get("recurrence-id"), settings)
        if recurrence_id:  # a moved/edited single occurrence of a series
            overrides.append((_event_id(url, uid), recurrence_id, fields))
            continue
        exdates = _exdates(component, settings)
        events[_event_id(url, uid)] = store.Event(
            id=_event_id(url, uid),
            rrule=_rrule(component),
            exdates=json.dumps(exdates) if exdates else "",
            **fields,
        )

    for master_id, occurrence, fields in overrides:
        master = events.get(master_id)
        if master is None:
            continue  # override without its master in the window — skip
        data = json.loads(master.overrides or "{}")
        data[occurrence] = {k: v for k, v in fields.items() if v}
        master.overrides = json.dumps(data)
    return events


def _differs(current: store.Event, incoming: store.Event) -> bool:
    return any(
        getattr(current, field) != getattr(incoming, field) for field in _MIRRORED_FIELDS
    )


def pull_feed(settings: Settings, url: str) -> dict:
    """Mirror one ICS feed into the store; return counts of what changed."""
    request = urllib.request.Request(url, headers={"User-Agent": "wakiru-assistant"})
    with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
        text = response.read(_MAX_FEED_BYTES).decode("utf-8", errors="replace")
    incoming = parse_feed(text, url, settings)

    prefix = _feed_prefix(url)
    existing = {e.id: e for e in store.list_events(settings) if e.id.startswith(prefix)}

    added = updated = 0
    stamp = datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")
    for event_id, event in incoming.items():
        current = existing.get(event_id)
        if current is None:
            event.created = event.updated = stamp
            store.restore_event(settings, event)
            added += 1
        elif _differs(current, event):
            event.created = current.created
            event.updated = stamp
            store.restore_event(settings, event)
            updated += 1

    removed = 0
    for event_id in existing:
        if event_id not in incoming:
            store.delete_event(settings, event_id)
            removed += 1

    return {"url": url, "events": len(incoming), "added": added, "updated": updated, "removed": removed}


def pull_feeds(settings: Settings | None = None) -> list[dict]:
    """Mirror every configured feed (best-effort per feed); return their stats."""
    settings = settings or get_settings()
    results = []
    for url in settings.calendar_ics_urls:
        try:
            result = pull_feed(settings, url)
        except Exception:
            logger.exception("calendar sync failed for feed %s…", url[:60])
            continue
        if result["added"] or result["updated"] or result["removed"]:
            logger.info(
                "calendar sync: +%d ~%d -%d (of %d) from %s…",
                result["added"], result["updated"], result["removed"],
                result["events"], url[:60],
            )
        results.append(result)
    return results
