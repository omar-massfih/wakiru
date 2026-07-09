"""IMAP/SMTP mailbox client — read, draft, and (gated) send.

Stdlib only (``imaplib`` / ``smtplib`` / ``email``), matching the rest of the
project's no-runtime-HTTP-dependency style. Authentication is XOAUTH2 with a
refreshed access token (:mod:`.oauth`) or a plain app password.

**Safety posture.** Nothing here runs unless ``enable_email`` is set. Listing and
reading use ``BODY.PEEK`` so the assistant never silently marks your mail as read.
Drafting appends to the drafts folder and is the default write. :func:`send_message`
is behind a *second*, independent switch (``enable_email_send``) and is never called
from any background path — only from an explicit user request.
"""

from __future__ import annotations

import imaplib
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr

from ..config import Settings
from .oauth import MailAuthError, access_token, xoauth2_string

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 20


@dataclass
class Message:
    """One mailbox message. ``body`` is populated only by :func:`read_message`."""

    uid: str
    sender: str
    subject: str
    date: str
    unread: bool = False
    body: str = ""


class MailDisabledError(RuntimeError):
    """Raised when an email operation is attempted while it is switched off."""


def _require_enabled(settings: Settings) -> None:
    if not settings.enable_email:
        raise MailDisabledError("Email is disabled. Set ENABLE_EMAIL=true to use it.")
    if not settings.email_address:
        raise MailAuthError("EMAIL_ADDRESS is not set.")


def _decode(raw: str | None) -> str:
    """Decode an RFC 2047 encoded header ('=?utf-8?B?...?=') to plain text."""
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def imap_connect(settings: Settings) -> imaplib.IMAP4_SSL:
    """Open and authenticate an IMAP connection. The seam tests monkeypatch."""
    _require_enabled(settings)
    conn = imaplib.IMAP4_SSL(
        settings.email_imap_host, settings.email_imap_port, timeout=_TIMEOUT_SECONDS
    )
    if settings.email_auth == "oauth":
        token = access_token(settings)
        auth = xoauth2_string(settings.email_address, token)
        conn.authenticate("XOAUTH2", lambda _: auth.encode())
    else:
        if not settings.email_password:
            raise MailAuthError("email_auth='password' requires EMAIL_PASSWORD.")
        conn.login(settings.email_address, settings.email_password)
    return conn


def _smtp_connect(settings: Settings) -> smtplib.SMTP:
    _require_enabled(settings)
    conn = smtplib.SMTP(
        settings.email_smtp_host, settings.email_smtp_port, timeout=_TIMEOUT_SECONDS
    )
    conn.starttls()
    if settings.email_auth == "oauth":
        token = access_token(settings)
        auth = xoauth2_string(settings.email_address, token)
        conn.auth("XOAUTH2", lambda _=None: auth, initial_response_ok=True)
    else:
        if not settings.email_password:
            raise MailAuthError("email_auth='password' requires EMAIL_PASSWORD.")
        conn.login(settings.email_address, settings.email_password)
    return conn


def list_recent(settings: Settings, unread_only: bool = True, limit: int | None = None) -> list[Message]:
    """Recent messages from INBOX, newest first. Headers only — ``BODY.PEEK`` keeps
    unread mail unread."""
    limit = limit or settings.email_max_messages
    conn = imap_connect(settings)
    try:
        conn.select("INBOX", readonly=True)
        criterion = "UNSEEN" if unread_only else "ALL"
        _, data = conn.uid("SEARCH", None, criterion)
        uids = (data[0] or b"").split()
        messages: list[Message] = []
        for uid in reversed(uids[-limit:]):  # newest first
            _, fetched = conn.uid(
                "FETCH", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
            )
            if not fetched or not isinstance(fetched[0], tuple):
                continue
            headers = message_from_bytes(fetched[0][1])
            messages.append(
                Message(
                    uid=uid.decode(),
                    sender=_decode(headers.get("From")),
                    subject=_decode(headers.get("Subject")),
                    date=_decode(headers.get("Date")),
                    unread=unread_only,
                )
            )
        return messages
    finally:
        _close(conn)


def read_message(settings: Settings, uid: str) -> Message | None:
    """Fetch one message with its plain-text body. Uses ``BODY.PEEK`` so reading it
    through the assistant does not mark it read in your mailbox."""
    conn = imap_connect(settings)
    try:
        conn.select("INBOX", readonly=True)
        _, fetched = conn.uid("FETCH", uid, "(BODY.PEEK[])")
        if not fetched or not isinstance(fetched[0], tuple):
            return None
        parsed = message_from_bytes(fetched[0][1])
        return Message(
            uid=uid,
            sender=_decode(parsed.get("From")),
            subject=_decode(parsed.get("Subject")),
            date=_decode(parsed.get("Date")),
            body=_plain_text_body(parsed),
        )
    finally:
        _close(conn)


def _plain_text_body(parsed) -> str:
    """The message's text/plain part (first one wins); '' when there is none."""
    if parsed.is_multipart():
        for part in parsed.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                payload = part.get_payload(decode=True) or b""
                return payload.decode(part.get_content_charset() or "utf-8", "replace")
        return ""
    payload = parsed.get_payload(decode=True) or b""
    return payload.decode(parsed.get_content_charset() or "utf-8", "replace")


def _require_address(to: str) -> str:
    """Validate a recipient. ``parseaddr`` only *parses* — it happily returns
    garbage unchanged — so check the shape before we ever address a message."""
    addr = parseaddr(to)[1]
    local, _, domain = addr.partition("@")
    if not (local and domain and "." in domain and " " not in addr):
        raise ValueError(f"{to!r} is not a valid email address")
    return addr


def _build(settings: Settings, to: str, subject: str, body: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = settings.email_address
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    return message


def save_draft(settings: Settings, to: str, subject: str, body: str) -> str:
    """Append a draft to the drafts folder. The default write path — never sends.

    Returns a short confirmation summary.
    """
    _require_address(to)
    message = _build(settings, to, subject, body)
    conn = imap_connect(settings)
    try:
        stamp = imaplib.Time2Internaldate(datetime.now(timezone.utc).timestamp())
        conn.append(settings.email_drafts_folder, r"\Draft", stamp, message.as_bytes())
    finally:
        _close(conn)
    return f"drafted: “{subject}” to {to}"


def send_message(settings: Settings, to: str, subject: str, body: str) -> str:
    """Send mail. Gated behind ``enable_email_send`` *in addition to* ``enable_email``.

    Never called from a background path — sending is irreversible, so it happens
    only when the user explicitly asks for it.
    """
    _require_enabled(settings)
    if not settings.enable_email_send:
        raise MailDisabledError(
            "Sending is disabled. I saved a draft instead — "
            "set ENABLE_EMAIL_SEND=true to allow sending."
        )
    _require_address(to)
    message = _build(settings, to, subject, body)
    conn = _smtp_connect(settings)
    try:
        conn.send_message(message)
    finally:
        try:
            conn.quit()
        except Exception:
            pass
    return f"sent: “{subject}” to {to}"


def _close(conn: imaplib.IMAP4_SSL) -> None:
    """Best-effort logout; a failed teardown must not mask the caller's result."""
    try:
        conn.logout()
    except Exception:
        logger.debug("imap logout failed", exc_info=True)
