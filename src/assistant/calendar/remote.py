"""Remote-calendar provider seam.

The two-way sync speaks to one of two transports — CalDAV (Fastmail/iCloud/Nextcloud)
or the Google Calendar REST API — selected by ``caldav_provider``. Both are expressed
here in ``store.Event`` terms so the shared machinery (:func:`sync.pull_caldav`,
:func:`ops._push_caldav`, :func:`sync.reconcile_caldav`) never branches on provider:

* :func:`list_events` — the remote's events as ``store.Event`` rows (id + caldav_href +
  caldav_etag set),
* :func:`upsert` — create or update one event, returning its ``(href, etag)``,
* :func:`delete` — remove one event by href/etag.

Provider-specific conflict/errors are normalized to :class:`RemoteError` /
:class:`RemoteConflictError` so callers handle them uniformly.
"""

from __future__ import annotations

from ..config import Settings
from . import store


class RemoteError(RuntimeError):
    """A remote-calendar operation failed (transport, auth, or bad status)."""


class RemoteConflictError(RemoteError):
    """A precondition failed — the remote changed since we last saw it."""


def is_google(settings: Settings) -> bool:
    return settings.caldav_provider == "google"


def is_configured(settings: Settings) -> bool:
    """Whether a remote calendar is set up enough to sync (reads need enable_caldav)."""
    if not settings.enable_caldav:
        return False
    if is_google(settings):
        return bool(settings.caldav_oauth_refresh_token)
    return bool(settings.caldav_url)


def list_events(settings: Settings) -> list[store.Event]:
    if is_google(settings):
        from . import google_calendar

        try:
            return google_calendar.list_events(settings)
        except google_calendar.GoogleCalError as exc:
            raise RemoteError(str(exc)) from exc
    from . import caldav

    try:
        return caldav.list_events(settings)
    except caldav.CalDavError as exc:
        raise RemoteError(str(exc)) from exc


def upsert(settings: Settings, event: store.Event) -> tuple[str, str]:
    """Create or update ``event`` remotely (create when it has no href yet); return
    the resource's ``(href, etag)``."""
    if is_google(settings):
        from . import google_calendar

        try:
            return google_calendar.upsert_event(settings, event)
        except google_calendar.GoogleConflictError as exc:
            raise RemoteConflictError(str(exc)) from exc
        except google_calendar.GoogleCalError as exc:
            raise RemoteError(str(exc)) from exc
    from . import caldav

    href = event.caldav_href or caldav.new_href(settings, event.id)
    ical = caldav.event_to_ical(settings, event)
    try:
        result = caldav.put_event(settings, href=href, ical=ical, etag=event.caldav_etag or None)
    except caldav.CalDavConflictError as exc:
        raise RemoteConflictError(str(exc)) from exc
    except caldav.CalDavError as exc:
        raise RemoteError(str(exc)) from exc
    return result.href, result.etag


def delete(settings: Settings, href: str, etag: str | None) -> None:
    if is_google(settings):
        from . import google_calendar

        try:
            google_calendar.delete_event(settings, href, etag)
            return
        except google_calendar.GoogleConflictError as exc:
            raise RemoteConflictError(str(exc)) from exc
        except google_calendar.GoogleCalError as exc:
            raise RemoteError(str(exc)) from exc
    from . import caldav

    try:
        caldav.delete_event(settings, href=href, etag=etag or None)
    except caldav.CalDavConflictError as exc:
        raise RemoteConflictError(str(exc)) from exc
    except caldav.CalDavError as exc:
        raise RemoteError(str(exc)) from exc
