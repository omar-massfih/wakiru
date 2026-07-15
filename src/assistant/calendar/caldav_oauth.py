"""OAuth2 access tokens for Google CalDAV (Bearer), stdlib-only.

Google does not allow password/Basic CalDAV, so its two-way sync uses OAuth2: a
long-lived *refresh* token (minted once, out of band, by
``scripts/caldav_oauth_setup.py``) is exchanged for short-lived *access* tokens at
Google's token endpoint. Access tokens are cached 0600 under the memory directory
with their expiry so a restart doesn't force a round-trip; the refresh token itself
is never written to disk here.

A near-twin of :mod:`assistant.mail.oauth` — same discipline, different settings and
cache path — kept separate rather than shared so the calendar's credential handling
doesn't couple to the mail subsystem.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request

from ..config import Settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
# Refresh a little early so a token can't expire mid-request.
_EXPIRY_SKEW_SECONDS = 60


def _load_cached(settings: Settings) -> str | None:
    path = settings.caldav_token_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    token, expires_at = data.get("access_token"), data.get("expires_at", 0)
    if token and time.time() < float(expires_at) - _EXPIRY_SKEW_SECONDS:
        return token
    return None


def _store_cached(settings: Settings, token: str, expires_in: int) -> None:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    payload = {"access_token": token, "expires_at": time.time() + expires_in}
    try:
        # The token grants calendar access for its lifetime, so it must never be
        # loose-permissioned, even briefly: create the file 0600 from the start and
        # publish it atomically (write_text-then-chmod leaves a umask-permissioned window).
        path = settings.caldav_token_path
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload))
        os.replace(tmp, path)
    except OSError:
        logger.warning("could not cache the caldav access token", exc_info=True)


def _refresh(settings: Settings) -> tuple[str, int]:
    from .caldav import CalDavAuthError

    missing = [
        name
        for name, value in (
            ("CALDAV_OAUTH_CLIENT_ID", settings.caldav_oauth_client_id),
            ("CALDAV_OAUTH_CLIENT_SECRET", settings.caldav_oauth_client_secret),
            ("CALDAV_OAUTH_REFRESH_TOKEN", settings.caldav_oauth_refresh_token),
        )
        if not value
    ]
    if missing:
        raise CalDavAuthError(f"caldav_auth='oauth' requires {', '.join(missing)}")

    body = urllib.parse.urlencode(
        {
            "client_id": settings.caldav_oauth_client_id,
            "client_secret": settings.caldav_oauth_client_secret,
            "refresh_token": settings.caldav_oauth_refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode()
    request = urllib.request.Request(
        settings.caldav_oauth_token_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read())
    except Exception as exc:  # urllib raises a zoo of errors; all mean "no token"
        raise CalDavAuthError(f"refreshing the caldav access token failed: {exc}") from exc

    token = data.get("access_token")
    if not token:
        raise CalDavAuthError("token endpoint returned no access_token")
    return token, int(data.get("expires_in", 3600))


def access_token(settings: Settings) -> str:
    """A valid OAuth2 access token, from cache when possible, else refreshed."""
    cached = _load_cached(settings)
    if cached:
        return cached
    token, expires_in = _refresh(settings)
    _store_cached(settings, token, expires_in)
    return token
