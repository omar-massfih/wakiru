"""OAuth2 access tokens for IMAP/SMTP (XOAUTH2), stdlib-only.

A long-lived *refresh* token (obtained once, out of band, and kept in the
environment) is exchanged for short-lived *access* tokens at the provider's token
endpoint. Access tokens are cached under the memory directory with their expiry
so a restart doesn't force a round-trip; the refresh token itself is never
written to disk by this module.
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
# Refresh a little early so a token can't expire mid-connection.
_EXPIRY_SKEW_SECONDS = 60


class MailAuthError(RuntimeError):
    """Raised when the mailbox cannot be authenticated."""


def _load_cached(settings: Settings) -> str | None:
    path = settings.mail_token_path
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
        # The token grants mailbox access for its lifetime, so it must never be
        # loose-permissioned, even briefly: create the file 0600 from the start
        # (write_text-then-chmod leaves a umask-permissioned window) and publish
        # it atomically.
        path = settings.mail_token_path
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload))
        os.replace(tmp, path)
    except OSError:
        logger.warning("could not cache the mail access token", exc_info=True)


def _refresh(settings: Settings) -> tuple[str, int]:
    missing = [
        name
        for name, value in (
            ("EMAIL_OAUTH_CLIENT_ID", settings.email_oauth_client_id),
            ("EMAIL_OAUTH_CLIENT_SECRET", settings.email_oauth_client_secret),
            ("EMAIL_OAUTH_REFRESH_TOKEN", settings.email_oauth_refresh_token),
        )
        if not value
    ]
    if missing:
        raise MailAuthError(f"email_auth='oauth' requires {', '.join(missing)}")

    body = urllib.parse.urlencode(
        {
            "client_id": settings.email_oauth_client_id,
            "client_secret": settings.email_oauth_client_secret,
            "refresh_token": settings.email_oauth_refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode()
    request = urllib.request.Request(
        settings.email_oauth_token_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read())
    except Exception as exc:  # urllib raises a zoo of errors; all mean "no token"
        raise MailAuthError(f"refreshing the mail access token failed: {exc}") from exc

    token = data.get("access_token")
    if not token:
        raise MailAuthError("token endpoint returned no access_token")
    return token, int(data.get("expires_in", 3600))


def access_token(settings: Settings) -> str:
    """A valid OAuth2 access token, from cache when possible, else refreshed."""
    cached = _load_cached(settings)
    if cached:
        return cached
    token, expires_in = _refresh(settings)
    _store_cached(settings, token, expires_in)
    return token


def xoauth2_string(email_address: str, token: str) -> str:
    """The SASL XOAUTH2 initial-response string IMAP/SMTP expect."""
    return f"user={email_address}\x01auth=Bearer {token}\x01\x01"
