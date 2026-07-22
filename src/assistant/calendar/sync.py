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
from datetime import date, datetime, timedelta

from .. import netguard
from ..config import Settings, get_settings
from . import store
from .context import resolve_tz

logger = logging.getLogger(__name__)

_SYNCED_PREFIX = "ics"
_CALDAV_PREFIX = "cdv"
_WAKIRU_UID_SUFFIX = "@wakiru"
_FETCH_TIMEOUT_SECONDS = 30
_MAX_FEED_BYTES = 10_000_000

# Fields that participate in change detection between pulls.
_MIRRORED_FIELDS = ("title", "start", "end", "location", "notes", "rrule", "exdates", "overrides")


def is_synced_id(event_id: str) -> bool:
    """Whether an event id belongs to a *read-only* ICS-mirrored external event.

    Deliberately ICS-only: CalDAV-backed rows (``cdv…`` ids, or our own writes
    carrying a ``caldav_href``) are read+**write**, so the write path must not
    refuse them — see :func:`assistant.calendar.ops._target_id`.
    """
    return event_id.startswith(_SYNCED_PREFIX)


def is_caldav_id(event_id: str) -> bool:
    """Whether an event id belongs to a *foreign* CalDAV event pulled from the collection."""
    return event_id.startswith(_CALDAV_PREFIX)


def _feed_prefix(url: str) -> str:
    return _SYNCED_PREFIX + hashlib.sha1(url.encode()).hexdigest()[:3]


def _event_id(url: str, uid: str) -> str:
    return _feed_prefix(url) + hashlib.sha1(uid.encode()).hexdigest()[:6]


def _caldav_local_id(uid: str) -> str:
    """The local id a pulled CalDAV event maps to.

    Our own writes carry ``UID = '<local id>@wakiru'``, so resolve straight back to
    that id — create → push → pull then upserts the same row instead of duplicating
    it. Foreign events get a stable ``cdv…`` id derived from their UID.
    """
    if uid.endswith(_WAKIRU_UID_SUFFIX):
        return uid[: -len(_WAKIRU_UID_SUFFIX)]
    return _CALDAV_PREFIX + hashlib.sha1(uid.encode()).hexdigest()[:9]


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


def parse_vevents(text: str, settings: Settings) -> dict[str, store.Event]:
    """Parse a VCALENDAR into events keyed by their **raw UID**, id left blank.

    Shared by the ICS pull and the CalDAV pull so both directions agree on tz and
    all-day handling; each caller assigns the final id (feed-scoped for ICS,
    ``cdv…`` for CalDAV — see :func:`_event_id` / :func:`assistant.calendar.caldav`).
    ``RECURRENCE-ID`` components fold into their master's ``overrides``.
    """
    import json

    from icalendar import Calendar

    calendar = Calendar.from_ical(text)
    events: dict[str, store.Event] = {}
    overrides: list[tuple[str, str, dict[str, str]]] = []  # (uid, occurrence, fields)

    for component in calendar.walk("VEVENT"):
        uid = _text(component, "uid")
        start = _iso(component.get("dtstart"), settings)
        if not uid or not start:
            continue
        end = _iso(component.get("dtend"), settings)
        if not end:
            dtstart = getattr(component.get("dtstart"), "dt", None)
            if isinstance(dtstart, date) and not isinstance(dtstart, datetime):
                # RFC 5545: DTEND is optional; a date-only DTSTART implies a
                # one-day event. Without this the store's 60-minute default
                # would leave 23 hours of an all-day event looking free.
                end = (
                    datetime(
                        dtstart.year, dtstart.month, dtstart.day,
                        tzinfo=resolve_tz(settings),
                    )
                    + timedelta(days=1)
                ).isoformat()
        fields = {
            "title": _text(component, "summary") or "(untitled)",
            "start": start,
            "end": end,
            "location": _text(component, "location"),
            "notes": _text(component, "description"),
        }
        recurrence_id = _iso(component.get("recurrence-id"), settings)
        if recurrence_id:  # a moved/edited single occurrence of a series
            overrides.append((uid, recurrence_id, fields))
            continue
        exdates = _exdates(component, settings)
        events[uid] = store.Event(
            id="",
            rrule=_rrule(component),
            exdates=json.dumps(exdates) if exdates else "",
            **fields,
        )

    for uid, occurrence, fields in overrides:
        master = events.get(uid)
        if master is None:
            continue  # override without its master in the window — skip
        data = json.loads(master.overrides or "{}")
        data[occurrence] = {k: v for k, v in fields.items() if v}
        master.overrides = json.dumps(data)
    return events


def parse_feed(text: str, url: str, settings: Settings) -> dict[str, store.Event]:
    """Parse ICS text into upsert-ready events keyed by their feed-derived id."""
    events: dict[str, store.Event] = {}
    for uid, event in parse_vevents(text, settings).items():
        event.id = _event_id(url, uid)
        events[event.id] = event
    return events


def _differs(current: store.Event, incoming: store.Event) -> bool:
    return any(
        getattr(current, field) != getattr(incoming, field) for field in _MIRRORED_FIELDS
    )


def pull_feed(settings: Settings, url: str) -> dict:
    """Mirror one ICS feed into the store; return counts of what changed.

    Feed URLs are operator-configured, but their redirects are not — the
    fetch goes through the SSRF guard, so a feed on a private host is
    refused (see :mod:`assistant.netguard`).
    """
    with netguard.urlopen_public(
        url,
        timeout=_FETCH_TIMEOUT_SECONDS,
        headers={"User-Agent": "wakiru-assistant"},
    ) as response:
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


def pull_caldav(settings: Settings) -> dict:
    """Mirror the writable remote calendar (CalDAV or Google) into read+write local rows.

    Unlike the ICS pull, these rows carry ``caldav_href``/``caldav_etag`` and are
    *not* refused by the write path — a later local edit is pushed back. A row with
    a queued outbox push (a local edit that hasn't reached the server) wins over the
    remote copy until reconcile settles it, so the pull never clobbers a pending edit.
    """
    from . import outbox, remote

    incoming: dict[str, store.Event] = {e.id: e for e in remote.list_events(settings)}

    # Remote-backed rows are exactly those carrying an href (foreign events and our
    # own successfully-pushed writes alike).
    existing = {e.id: e for e in store.list_events(settings) if e.caldav_href}

    added = updated = skipped = 0
    stamp = datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")
    for event_id, event in incoming.items():
        if outbox.has_pending(settings, event_id):
            skipped += 1
            continue
        current = existing.get(event_id)
        if current is None:
            event.created = event.updated = stamp
            store.restore_event(settings, event)
            added += 1
        elif current.caldav_etag != event.caldav_etag or _differs(current, event):
            event.created = current.created or stamp
            event.updated = stamp
            store.restore_event(settings, event)
            updated += 1

    removed = 0
    for event_id in existing:
        if event_id not in incoming and not outbox.has_pending(settings, event_id):
            store.delete_event(settings, event_id)
            removed += 1

    if added or updated or removed:
        logger.info(
            "caldav sync: +%d ~%d -%d (of %d) from the collection",
            added, updated, removed, len(incoming),
        )
    return {
        "provider": settings.caldav_provider,
        "events": len(incoming),
        "added": added,
        "updated": updated,
        "removed": removed,
        "skipped": skipped,
    }


def reconcile_caldav(settings: Settings) -> dict:
    """Retry every queued CalDAV push (the outbox), draining what a prior outage left.

    The only background path allowed to touch the remote, and only to flush writes the
    user already intended. A push that keeps conflicting (412) resolves *toward the
    remote* — the pending intent is dropped and the next pull overwrites local — so an
    interactive edit that lost a race never silently stomps a newer remote change.
    """
    from . import outbox, remote

    reconciled = dropped = still_pending = 0
    for row in outbox.pending(settings):
        event_id = row["event_id"]
        try:
            if row["op"] == outbox.OP_DELETE:
                remote.delete(settings, row["href"], row["etag"] or None)
            else:
                event = store.get_event(settings, event_id)
                if event is None:  # created then deleted before we could push — moot
                    outbox.clear(settings, event_id)
                    dropped += 1
                    continue
                href, etag = remote.upsert(settings, event)
                store.set_caldav_meta(settings, event_id, href, etag)
            outbox.clear(settings, event_id)
            reconciled += 1
        except remote.RemoteConflictError:
            outbox.clear(settings, event_id)  # remote wins; the next pull fixes local
            dropped += 1
            logger.warning(
                "caldav reconcile: %s %s conflicts remotely; deferring to the server",
                row["op"], event_id,
            )
        except remote.RemoteError:
            still_pending += 1  # still offline — leave it queued for the next pass
        except Exception:
            # An unexpected error on one row — a corrupt event, a DB hiccup, a
            # provider error not wrapped as RemoteError, an iCal-build failure —
            # must not abort the whole drain and strand every push queued behind
            # it. Isolate per row like pull_feeds does; leave it queued to retry.
            still_pending += 1
            logger.exception(
                "caldav reconcile: unexpected error on %s %s; leaving it queued",
                row["op"], event_id,
            )

    return {"reconciled": reconciled, "dropped": dropped, "still_pending": still_pending}


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
