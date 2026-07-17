"""Google Calendar REST API v3 transport — the "google" remote provider.

Google walls off CalDAV (v1 is dead, v2 returns 403 to ordinary OAuth apps), but the
Calendar REST API works with the same OAuth token. This module speaks that API in the
same shape the CalDAV client exposes — list the calendar into ``store.Event`` rows,
and create/update/delete events — so :mod:`assistant.calendar.remote` can dispatch to
either transport and the pull/push/undo/outbox machinery stays shared.

Stdlib only (a ``_request`` seam for offline tests, routed through the SSRF guard),
OAuth2 Bearer via :mod:`assistant.calendar.caldav_oauth`. Our own creates set the
Google event id to the local event id (a 12-char hex string, a valid Google id), so a
later pull recognizes them and upserts in place instead of duplicating.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, timedelta

from .. import netguard
from ..config import Settings
from . import store

logger = logging.getLogger(__name__)

_API = "https://www.googleapis.com/calendar/v3"
_TIMEOUT_SECONDS = 30
_MAX_BODY_BYTES = 20_000_000
_MAX_REDIRECTS = 5
_PAGE_SIZE = 2500
_EVENT_COLOR_IDS = tuple(str(value) for value in range(1, 12))


class GoogleCalError(RuntimeError):
    """A Google Calendar request failed (bad status or transport error)."""


class GoogleAuthError(GoogleCalError):
    """Credentials rejected (401/403)."""


class GoogleConflictError(GoogleCalError):
    """An ``If-Match`` precondition failed (412/409) — the remote moved."""


# --- the network seam ----------------------------------------------------------


def _request(
    method: str,
    url: str,
    *,
    settings: Settings,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One REST round-trip → ``(status, lowercased_headers, body)``. THE test seam.

    Bearer auth, SSRF-guarded with per-hop redirect re-validation, never raises on a
    4xx/5xx (the status is returned so callers map it); only transport failure raises.
    """
    from .caldav_oauth import access_token

    hdrs = {"Authorization": f"Bearer {access_token(settings)}", **(headers or {})}
    for _ in range(_MAX_REDIRECTS + 1):
        netguard.require_public_url(url)
        request = urllib.request.Request(url, data=body or None, method=method, headers=hdrs)
        opener = urllib.request.build_opener(netguard._StopRedirects)
        try:
            response = opener.open(request, timeout=_TIMEOUT_SECONDS)
        except netguard._RedirectSignal as signal:
            url = urllib.parse.urljoin(url, signal.target)
            continue
        except urllib.error.HTTPError as exc:
            return exc.code, _lower_headers(exc.headers), exc.read(_MAX_BODY_BYTES)
        except OSError as exc:
            raise GoogleCalError(f"Google {method} {url} failed: {exc}") from exc
        with response:
            status = getattr(response, "status", 0) or response.getcode()
            return status, _lower_headers(response.headers), response.read(_MAX_BODY_BYTES)
    raise GoogleCalError(f"too many redirects for {url}")


def _lower_headers(headers) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers.items() if headers else [])}


def _base(settings: Settings) -> str:
    cal = settings.google_calendar_id or "primary"
    return f"{_API}/calendars/{urllib.parse.quote(cal, safe='')}/events"


def _raise_for_status(status: int, method: str, url: str, body: bytes = b"") -> None:
    if status in (401, 403):
        raise GoogleAuthError(f"Google {method} {url} rejected the credentials ({status})")
    if status in (409, 412):
        raise GoogleConflictError(f"Google {method} {url} precondition failed ({status})")
    if status >= 400:
        detail = body.decode(errors="replace")[:200]
        raise GoogleCalError(f"Google {method} {url} returned {status}: {detail}")


# --- time mapping (store ISO <-> Google's start/end objects) -------------------


def _resolve_tz(settings: Settings):
    from .context import resolve_tz

    return resolve_tz(settings)


def _gtime_to_iso(obj: dict | None, settings: Settings) -> str:
    """A Google start/end object → a tz-aware ISO string ('' if absent).

    ``dateTime`` carries its own offset; a date-only (all-day) value becomes local
    midnight, matching the ICS pull's :func:`sync._iso`.
    """
    if not obj:
        return ""
    if obj.get("dateTime"):
        dt = store.parse_dt(str(obj["dateTime"]))
        return dt.isoformat(timespec="seconds") if dt else str(obj["dateTime"])
    if obj.get("date"):
        from datetime import datetime

        try:
            year, month, day = (int(part) for part in str(obj["date"]).split("-"))
            return datetime(year, month, day, tzinfo=_resolve_tz(settings)).isoformat()
        except ValueError:
            return ""
    return ""


def _iso_to_gtime(value: str) -> dict:
    dt = store.parse_dt(value)
    return {"dateTime": dt.isoformat(timespec="seconds")} if dt else {"dateTime": value}


def _color_id(event_id: str) -> str:
    """A random-looking Google palette color that stays stable for this event."""
    digest = hashlib.sha256(event_id.encode()).digest()
    return _EVENT_COLOR_IDS[int.from_bytes(digest[:8], "big") % len(_EVENT_COLOR_IDS)]


def _utc_basic(value: str) -> str:
    dt = store.parse_dt(value)
    if dt is None:
        return ""
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


# --- store.Event <-> Google event JSON -----------------------------------------


def _to_body(settings: Settings, event: store.Event) -> dict:
    body: dict = {"summary": event.title, "colorId": _color_id(event.id)}
    body["start"] = _iso_to_gtime(event.start)
    end = event.end
    if not end:
        dt = store.parse_dt(event.start)
        end = (dt + timedelta(hours=1)).isoformat(timespec="seconds") if dt else event.start
    body["end"] = _iso_to_gtime(end)
    if event.location:
        body["location"] = event.location
    if event.notes:
        body["description"] = event.notes
    recurrence: list[str] = []
    if event.rrule:
        recurrence.append("RRULE:" + event.rrule)
    exdates = [d for d in (_utc_basic(x) for x in store.load_exdates(event)) if d]
    if exdates:
        recurrence.append("EXDATE:" + ",".join(exdates))
    if recurrence:
        body["recurrence"] = recurrence
    return body


def _from_google(g: dict, settings: Settings) -> store.Event:
    event = store.Event(
        id=str(g["id"]),
        title=str(g.get("summary") or "(untitled)"),
        start=_gtime_to_iso(g.get("start"), settings),
        end=_gtime_to_iso(g.get("end"), settings),
        location=str(g.get("location") or ""),
        notes=str(g.get("description") or ""),
        caldav_href=str(g["id"]),
        caldav_etag=str(g.get("etag") or ""),
    )
    exdates: list[str] = []
    for line in g.get("recurrence") or []:
        if line.startswith("RRULE:"):
            event.rrule = line[len("RRULE:"):]
        elif line.startswith("EXDATE"):
            _, _, values = line.partition(":")
            for chunk in values.split(","):
                dt = store.parse_dt(chunk.strip())
                if dt is not None:
                    exdates.append(dt.isoformat(timespec="seconds"))
    if exdates:
        event.exdates = json.dumps(exdates)
    return event


# --- provider operations -------------------------------------------------------


def list_events(settings: Settings) -> list[store.Event]:
    """Mirror the calendar into ``store.Event`` rows (masters; instance edits folded)."""
    masters: dict[str, store.Event] = {}
    instances: list[dict] = []
    page_token: str | None = None
    while True:
        params = {"singleEvents": "false", "maxResults": str(_PAGE_SIZE), "showDeleted": "false"}
        if page_token:
            params["pageToken"] = page_token
        url = _base(settings) + "?" + urllib.parse.urlencode(params)
        status, _, body = _request("GET", url, settings=settings)
        _raise_for_status(status, "GET", url, body)
        data = json.loads(body)
        for g in data.get("items", []):
            if g.get("status") == "cancelled" or not g.get("id"):
                continue
            if g.get("recurringEventId"):
                instances.append(g)
                continue
            event = _from_google(g, settings)
            masters[event.id] = event
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    for g in instances:  # fold moved/edited single occurrences into their master
        master = masters.get(str(g.get("recurringEventId")))
        occurrence = _gtime_to_iso(g.get("originalStartTime"), settings)
        if master is None or not occurrence:
            continue
        overrides = json.loads(master.overrides or "{}")
        overrides[occurrence] = {
            k: v
            for k, v in {
                "title": g.get("summary"),
                "start": _gtime_to_iso(g.get("start"), settings),
                "end": _gtime_to_iso(g.get("end"), settings),
                "location": g.get("location"),
            }.items()
            if v
        }
        master.overrides = json.dumps(overrides)
    return list(masters.values())


def upsert_event(settings: Settings, event: store.Event) -> tuple[str, str]:
    """Create or update the event; return its (id, etag). Update falls back to a
    recreate if the remote resource is gone (e.g. undoing a cancel)."""
    body = _to_body(settings, event)
    if event.caldav_href:
        url = _base(settings) + "/" + urllib.parse.quote(event.caldav_href, safe="")
        headers = {"Content-Type": "application/json"}
        if event.caldav_etag:
            headers["If-Match"] = event.caldav_etag
        status, _, resp = _request(
            "PUT", url, settings=settings, body=json.dumps(body).encode(), headers=headers
        )
        if status in (404, 410):
            return _insert(settings, event, body)
        _raise_for_status(status, "PUT", url, resp)
        g = json.loads(resp)
        return str(g["id"]), str(g.get("etag") or "")
    return _insert(settings, event, body)


def _insert(settings: Settings, event: store.Event, body: dict) -> tuple[str, str]:
    payload = {**body, "id": event.id}  # our id so a pull recognizes our own event
    url = _base(settings)
    status, _, resp = _request(
        "POST", url, settings=settings, body=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if status == 409:  # already exists (a retried create) — overwrite it in place
        put_url = _base(settings) + "/" + urllib.parse.quote(event.id, safe="")
        status, _, resp = _request(
            "PUT", put_url, settings=settings, body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    _raise_for_status(status, "POST", url, resp)
    g = json.loads(resp)
    return str(g["id"]), str(g.get("etag") or "")


def delete_event(settings: Settings, href: str, etag: str | None) -> None:
    """Delete an event by id; a 404/410 (already gone) is treated as success."""
    url = _base(settings) + "/" + urllib.parse.quote(href, safe="")
    headers = {"If-Match": etag} if etag else {}
    status, _, body = _request("DELETE", url, settings=settings, headers=headers)
    if status in (200, 204, 404, 410):
        return
    _raise_for_status(status, "DELETE", url, body)
    raise GoogleCalError(f"Google DELETE {url} returned {status}")
