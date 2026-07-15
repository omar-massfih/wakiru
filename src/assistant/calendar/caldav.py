"""A minimal, stdlib-only CalDAV client — the write half of two-way sync.

Where :mod:`assistant.calendar.sync` *reads* external calendars over a plain ICS
subscription, this module talks the CalDAV protocol to one **writable** collection:
it ``REPORT``s the collection to mirror it in (with each resource's href + ETag),
and ``PUT``s/``DELETE``s to push local writes back. Identity/state travels on the
event row's ``caldav_href``/``caldav_etag`` columns; the ETag is the ``If-Match``
precondition that makes a conflicting remote change fail loudly instead of being
silently clobbered.

Design, matching the rest of the project:

* No runtime HTTP dependency — stdlib :mod:`urllib.request`, the same as
  :mod:`assistant.mail`. Every request runs through the SSRF guard
  (:func:`assistant.netguard.require_public_url`) and re-validates redirects.
* One network seam, :func:`_request`, so tests monkeypatch a single function and
  assert the exact PUT/DELETE payloads offline — as the ICS tests do with
  ``netguard._open``.
* Basic auth over HTTPS (an app-specific password). ``caldav_auth='oauth'`` is a
  reserved future slot that would mirror :mod:`assistant.mail.oauth`.
"""

from __future__ import annotations

import base64
import logging
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime

from .. import netguard
from ..config import Settings
from . import store

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30
_MAX_BODY_BYTES = 10_000_000
_MAX_REDIRECTS = 5

# XML namespaces used in CalDAV multistatus responses.
_DAV = "DAV:"
_CALDAV = "urn:ietf:params:xml:ns:caldav"

_REPORT_BODY = (
    b'<?xml version="1.0" encoding="utf-8"?>'
    b'<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">'
    b"<D:prop><D:getetag/><C:calendar-data/></D:prop>"
    b"<C:filter><C:comp-filter name=\"VCALENDAR\">"
    b"<C:comp-filter name=\"VEVENT\"/>"
    b"</C:comp-filter></C:filter>"
    b"</C:calendar-query>"
)


class CalDavError(RuntimeError):
    """A CalDAV request failed (bad status, transport error, or misconfiguration)."""


class CalDavAuthError(CalDavError):
    """The server rejected the credentials (401/403)."""


class CalDavConflictError(CalDavError):
    """An ``If-Match``/``If-None-Match`` precondition failed (412/409) — the remote moved."""


@dataclass
class RemoteResource:
    href: str  # server path, e.g. "/dav/cal/home/abcd.ics"
    etag: str  # opaque validator, surrounding quotes stripped
    ical: str  # the resource's VCALENDAR body


@dataclass
class PutResult:
    href: str
    etag: str  # '' when the server withheld the ETag on PUT (re-fetched next pull)


# --- configuration + auth ------------------------------------------------------


def is_configured(settings: Settings) -> bool:
    """Whether a writable collection URL is set (reads still need enable_caldav)."""
    return bool(settings.caldav_url)


def _require_url(settings: Settings) -> str:
    url = settings.caldav_url
    if not url:
        raise CalDavError("CALDAV_URL is not set")
    return url if url.endswith("/") else url + "/"


def _auth_header(settings: Settings) -> dict[str, str]:
    if settings.caldav_auth == "oauth":
        # Google CalDAV: a Bearer access token, refreshed from the stored refresh token
        # (raises CalDavAuthError if the credentials are missing or the exchange fails).
        from .caldav_oauth import access_token

        return {"Authorization": f"Bearer {access_token(settings)}"}
    raw = f"{settings.caldav_username or ''}:{settings.caldav_password or ''}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}


# --- the single network seam ---------------------------------------------------


def _request(
    method: str,
    url: str,
    *,
    settings: Settings,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One CalDAV round-trip → ``(status, lowercased_headers, body)``.

    THE monkeypatch seam. Runs the SSRF guard first and re-validates every redirect
    hop (reusing :class:`netguard._StopRedirects`). Never raises on a 4xx/5xx — the
    status is returned so callers map it to the right :class:`CalDavError`; only a
    transport failure raises (as :class:`CalDavError`).
    """
    hdrs = {**_auth_header(settings), **(headers or {})}
    for _ in range(_MAX_REDIRECTS + 1):
        netguard.require_public_url(url)
        request = urllib.request.Request(
            url, data=body or None, method=method, headers=hdrs
        )
        opener = urllib.request.build_opener(netguard._StopRedirects)
        try:
            response = opener.open(request, timeout=_TIMEOUT_SECONDS)
        except netguard._RedirectSignal as signal:
            url = urllib.parse.urljoin(url, signal.target)
            continue
        except urllib.error.HTTPError as exc:
            return exc.code, _lower_headers(exc.headers), exc.read(_MAX_BODY_BYTES)
        except OSError as exc:
            raise CalDavError(f"CalDAV {method} {url} failed: {exc}") from exc
        with response:
            status = getattr(response, "status", 0) or response.getcode()
            return status, _lower_headers(response.headers), response.read(_MAX_BODY_BYTES)
    raise CalDavError(f"too many redirects for {url}")


def _lower_headers(headers) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers.items() if headers else [])}


def _strip_etag(value: str) -> str:
    return value.strip().strip('"') if value else ""


def _raise_for_status(status: int, method: str, href: str) -> None:
    if status in (401, 403):
        raise CalDavAuthError(f"CalDAV {method} {href} rejected the credentials ({status})")
    if status in (409, 412):
        raise CalDavConflictError(href, f"CalDAV {method} {href} precondition failed ({status})")
    if status >= 400:
        raise CalDavError(f"CalDAV {method} {href} returned {status}")


# --- protocol operations -------------------------------------------------------


def list_resources(settings: Settings) -> list[RemoteResource]:
    """``REPORT`` the collection → every VEVENT resource with its href, ETag, and body."""
    url = _require_url(settings)
    status, _, body = _request(
        "REPORT",
        url,
        settings=settings,
        body=_REPORT_BODY,
        headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
    )
    if status != 207:
        _raise_for_status(status, "REPORT", url)
        raise CalDavError(f"CalDAV REPORT {url} returned {status}, expected 207")
    return _parse_multistatus(body)


def _parse_multistatus(body: bytes) -> list[RemoteResource]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise CalDavError(f"malformed CalDAV multistatus: {exc}") from exc
    out: list[RemoteResource] = []
    for response in root.findall(f"{{{_DAV}}}response"):
        href = (response.findtext(f"{{{_DAV}}}href") or "").strip()
        etag = _strip_etag(response.findtext(f".//{{{_DAV}}}getetag") or "")
        ical = (response.findtext(f".//{{{_CALDAV}}}calendar-data") or "").strip()
        if href and ical:
            out.append(RemoteResource(href=href, etag=etag, ical=ical))
    return out


def resource_url(settings: Settings, href: str) -> str:
    """Resolve a possibly-relative resource href against the collection base."""
    return urllib.parse.urljoin(_require_url(settings), href)


def new_href(settings: Settings, event_id: str) -> str:
    """The resource URL a freshly-created local event is PUT to (chosen, not assigned)."""
    return urllib.parse.urljoin(_require_url(settings), f"{event_id}.ics")


def put_event(settings: Settings, *, href: str, ical: str, etag: str | None) -> PutResult:
    """``PUT`` a VCALENDAR. New resource → ``If-None-Match: *``; update → ``If-Match: <etag>``.

    A 412 (someone changed the remote first) raises :class:`CalDavConflictError`.
    """
    url = resource_url(settings, href)
    headers = {"Content-Type": "text/calendar; charset=utf-8"}
    headers["If-Match" if etag else "If-None-Match"] = f'"{etag}"' if etag else "*"
    status, resp_headers, _ = _request(
        "PUT", url, settings=settings, body=ical.encode(), headers=headers
    )
    _raise_for_status(status, "PUT", url)
    if status not in (200, 201, 204):
        raise CalDavError(f"CalDAV PUT {url} returned {status}")
    return PutResult(href=url, etag=_strip_etag(resp_headers.get("etag", "")))


def delete_event(settings: Settings, *, href: str, etag: str | None) -> None:
    """``DELETE`` a resource (``If-Match: <etag>`` when known). A 404 is treated as done."""
    url = resource_url(settings, href)
    headers = {"If-Match": f'"{etag}"'} if etag else {}
    status, _, _ = _request("DELETE", url, settings=settings, headers=headers)
    if status == 404:
        return  # already gone — the outcome we wanted
    _raise_for_status(status, "DELETE", url)
    if status not in (200, 204):
        raise CalDavError(f"CalDAV DELETE {url} returned {status}")


# --- VEVENT generation (the inverse of sync.parse_vevents) ---------------------


def event_to_ical(settings: Settings, event: store.Event) -> str:
    """Render a :class:`store.Event` as a single-resource VCALENDAR string.

    ``UID = f'{event.id}@wakiru'`` so a later pull recognizes our own resource.
    A recurring series emits the master VEVENT (RRULE + EXDATE) plus one extra
    VEVENT per ``RECURRENCE-ID`` override, all sharing the UID.
    """
    from icalendar import Calendar

    cal = Calendar()
    cal.add("prodid", "-//wakiru//caldav//EN")
    cal.add("version", "2.0")
    cal.add_component(_build_vevent(event, recurrence_id=None, fields=None))
    for occurrence, fields in store.load_overrides(event).items():
        cal.add_component(_build_vevent(event, recurrence_id=occurrence, fields=fields))
    return cal.to_ical().decode()


def _as_utc(value: str):
    dt = store.parse_dt(value)
    return dt.astimezone(UTC) if dt else None


def _build_vevent(event: store.Event, *, recurrence_id: str | None, fields: dict | None):
    from icalendar import Event as IEvent
    from icalendar.prop import vRecur

    fields = fields or {}
    ve = IEvent()
    ve.add("uid", f"{event.id}@wakiru")
    ve.add("dtstamp", datetime.now(UTC))

    if recurrence_id is not None:
        occ = _as_utc(recurrence_id)
        if occ is not None:
            ve.add("recurrence-id", occ)
        start = fields.get("start") or recurrence_id
    else:
        start = event.start

    ve.add("summary", fields.get("title") or event.title)
    dtstart = _as_utc(start)
    if dtstart is not None:
        ve.add("dtstart", dtstart)
    dtend = _as_utc(fields.get("end") or event.end)
    if dtend is not None:
        ve.add("dtend", dtend)
    location = fields.get("location") or event.location
    if location:
        ve.add("location", location)
    notes = fields.get("notes") or event.notes
    if notes:
        ve.add("description", notes)

    if recurrence_id is None and event.rrule:
        try:
            ve.add("rrule", vRecur.from_ical(event.rrule))
        except (ValueError, TypeError):
            logger.warning("dropping unparseable rrule %r on caldav write", event.rrule)
        for exdate in store.load_exdates(event):
            exdt = _as_utc(exdate)
            if exdt is not None:
                ve.add("exdate", exdt)

    ve.add("sequence", 0)
    return ve
