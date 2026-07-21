"""IMAP/SMTP mailbox client — read, draft, reply, manage, and (gated) send.

Stdlib only (``imaplib`` / ``smtplib`` / ``email``), matching the rest of the
project's no-runtime-HTTP-dependency style. Authentication is XOAUTH2 with a
refreshed access token (:mod:`.oauth`) or a plain app password.

**Safety posture.** Nothing here runs unless ``enable_email`` is set. Listing and
reading use ``BODY.PEEK`` so the assistant never silently marks your mail as read —
marking read is its own deliberate operation (:func:`mark_read`), never a side
effect of reading. The mailbox mutations (:func:`archive_message`, :func:`set_label`,
:func:`mark_read`) are chosen to be recoverable by hand: archiving keeps the message
(Gmail: All Mail; elsewhere: the archive folder), and labels and read state have
reverse verbs. Drafting (:func:`save_draft`, :func:`save_reply_draft`) appends to
the drafts folder and is the default write. :func:`send_message` / :func:`send_reply`
are behind a *second*, independent switch (``enable_email_send``) and are never
called from any background path — only from an explicit user request.
"""

from __future__ import annotations

import contextlib
import imaplib
import logging
import re
import smtplib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.message import Message as MimePart
from email.utils import getaddresses, make_msgid, parseaddr

from ..config import Settings
from .oauth import MailAuthError, access_token, xoauth2_string

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 20


@dataclass
class Message:
    """One mailbox message. ``body`` and ``attachments`` (filenames) are
    populated only by :func:`read_message`."""

    uid: str
    sender: str
    subject: str
    date: str
    unread: bool = False
    body: str = ""
    attachments: list[str] = field(default_factory=list)


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
    if not settings.email_address:
        raise MailAuthError("email requires EMAIL_ADDRESS.")
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
    if not settings.email_address:
        raise MailAuthError("email requires EMAIL_ADDRESS.")
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


def _headers_for(conn, uids: list[bytes], limit: int, unread: bool) -> list[Message]:
    """Header-only :class:`Message` rows for the newest ``limit`` uids —
    ``BODY.PEEK`` keeps unread mail unread."""
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
                unread=unread,
            )
        )
    return messages


def list_recent(settings: Settings, unread_only: bool = True, limit: int | None = None) -> list[Message]:
    """Recent messages from INBOX, newest first. Headers only — ``BODY.PEEK`` keeps
    unread mail unread."""
    limit = limit or settings.email_max_messages
    conn = imap_connect(settings)
    try:
        conn.select("INBOX", readonly=True)
        criterion = "UNSEEN" if unread_only else "ALL"
        # None is the CHARSET arg (valid for SEARCH); typeshed types uid's
        # varargs as str-only, so it can't see that.
        _, data = conn.uid("SEARCH", None, criterion)  # type: ignore[arg-type]
        uids = (data[0] or b"").split()
        return _headers_for(conn, uids, limit, unread=unread_only)
    finally:
        _close(conn)


# IMAP date syntax wants English month abbreviations; strftime's %b follows
# LC_TIME and would emit e.g. "jul." under a Norwegian locale — invalid syntax.
_IMAP_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _quote_criterion(value: str) -> str:
    """An IMAP SEARCH string argument, quoted.

    Embedded quotes/backslashes are escaped; CR/LF/NUL are replaced with
    spaces — imaplib transmits args verbatim, so a newline in a search term
    would otherwise inject a second command into the IMAP stream.
    """
    value = re.sub(r"[\r\n\x00]", " ", value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def search_messages(
    settings: Settings,
    sender: str = "",
    subject: str = "",
    text: str = "",
    since_days: int = 0,
    limit: int | None = None,
) -> list[Message]:
    """INBOX messages matching every given criterion, newest first.

    Maps straight onto IMAP ``SEARCH`` (server-side, so old mail is found
    without paging the whole mailbox down): ``FROM``/``SUBJECT``/``TEXT`` plus
    an optional ``SINCE`` cutoff. Read-only like :func:`list_recent` — the
    select is readonly and only headers are peeked. Non-ASCII terms go as
    ``CHARSET UTF-8`` with the query passed as *bytes* (imaplib would raise
    encoding a str); RFC 3501 strictly wants literals there, so acceptance of
    inline UTF-8 is server-dependent (Gmail and Dovecot take it) and a refusal
    surfaces as the server's error.
    """
    criteria: list[str] = []
    if sender.strip():
        criteria += ["FROM", _quote_criterion(sender.strip())]
    if subject.strip():
        criteria += ["SUBJECT", _quote_criterion(subject.strip())]
    if text.strip():
        criteria += ["TEXT", _quote_criterion(text.strip())]
    if since_days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=since_days)
        criteria += [
            "SINCE",
            f"{cutoff.day:02d}-{_IMAP_MONTHS[cutoff.month - 1]}-{cutoff.year}",
        ]
    if not criteria:
        return []
    query = "(" + " ".join(criteria) + ")"
    limit = limit or settings.email_max_messages
    conn = imap_connect(settings)
    try:
        conn.select("INBOX", readonly=True)
        if query.isascii():
            _, data = conn.uid("SEARCH", query)
        else:
            # typeshed wants str, but imaplib ASCII-encodes str args (raising
            # on these) while transmitting bytes untouched — bytes is the point.
            _, data = conn.uid("SEARCH", "CHARSET", "UTF-8", query.encode("utf-8"))  # type: ignore[arg-type]
        uids = (data[0] or b"").split()
        return _headers_for(conn, uids, limit, unread=False)
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
        return _parse_message(uid, parsed)
    finally:
        _close(conn)


def _parse_message(uid: str, parsed) -> Message:
    """A full :class:`Message` (body + attachment names) from a parsed fetch."""
    return Message(
        uid=uid,
        sender=_decode(parsed.get("From")),
        subject=_decode(parsed.get("Subject")),
        date=_decode(parsed.get("Date")),
        body=_plain_text_body(parsed),
        attachments=[name for name, _ in _attachment_parts(parsed)],
    )


def _attachment_parts(parsed) -> list[tuple[str, MimePart]]:
    """(decoded filename, part) for every part carrying a filename — the
    standard attachment signal (named inline parts count too; they are just as
    ingestable)."""
    return [
        (_decode(filename), part)
        for part in parsed.walk()
        if (filename := part.get_filename())
    ]


def read_with_attachment(
    settings: Settings, uid: str, name: str = ""
) -> tuple[Message | None, tuple[str, bytes] | None]:
    """One fetch, two answers: the message (as :func:`read_message`) and one
    resolved attachment's ``(filename, content)``.

    ``name`` matches the filename case-insensitively — a *unique* exact match
    first, then a *unique* substring; an empty ``name`` means the message's
    only attachment. The attachment half is ``None`` when nothing matches
    unambiguously (the caller can name the options from
    :attr:`Message.attachments`). One ``BODY.PEEK[]`` round-trip serves both —
    the message can't change between "list the attachments" and "take this
    one", and reading stays unread. The whole message is buffered either way
    (imaplib offers no partial fetch worth the BODYSTRUCTURE gymnastics), so
    any size cap is the caller's to apply to the returned bytes.
    """
    conn = imap_connect(settings)
    try:
        conn.select("INBOX", readonly=True)
        _, fetched = conn.uid("FETCH", uid, "(BODY.PEEK[])")
        if not fetched or not isinstance(fetched[0], tuple):
            return None, None
        parsed = message_from_bytes(fetched[0][1])
    finally:
        _close(conn)
    parts = _attachment_parts(parsed)
    message = _parse_message(uid, parsed)
    needle = name.strip().lower()
    if not needle:
        matches = parts if len(parts) == 1 else []
    else:
        exact = [(f, p) for f, p in parts if f.lower() == needle]
        if exact:
            # Two parts with the same filename stay ambiguous — never pick one
            # silently.
            matches = exact if len(exact) == 1 else []
        else:
            substring = [(f, p) for f, p in parts if needle in f.lower()]
            matches = substring if len(substring) == 1 else []
    if not matches:
        return message, None
    filename, part = matches[0]
    payload = part.get_payload(decode=True)
    return message, (filename, payload if isinstance(payload, bytes) else b"")


def fetch_attachment(
    settings: Settings, uid: str, name: str = ""
) -> tuple[str, bytes] | None:
    """Just the attachment half of :func:`read_with_attachment`."""
    _, attachment = read_with_attachment(settings, uid, name)
    return attachment


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
    # A Message-ID here means a draft that is later sent (from any client)
    # threads correctly; servers keep a supplied id rather than minting one.
    domain = (settings.email_address or "").partition("@")[2]
    message["Message-ID"] = make_msgid(domain=domain or None)
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
        stamp = imaplib.Time2Internaldate(datetime.now(UTC).timestamp())
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
        # Nothing has been written at this point — say so rather than claiming a
        # draft was saved. The caller can call save_draft() if it wants one.
        raise MailDisabledError(
            "Sending is disabled, so nothing was sent or drafted. "
            "Use a draft instead, or set ENABLE_EMAIL_SEND=true to allow sending."
        )
    _require_address(to)
    message = _build(settings, to, subject, body)
    conn = _smtp_connect(settings)
    try:
        conn.send_message(message)
    finally:
        with contextlib.suppress(Exception):
            conn.quit()
    return f"sent: “{subject}” to {to}"


# --------------------------------------------------------------------------- #
# Replies — threading headers are fetched server-side, never via the model
# --------------------------------------------------------------------------- #

# Subjects already carrying a reply prefix (English, Norwegian, German) are not
# prefixed again.
_REPLY_PREFIX = re.compile(r"^\s*(re|sv|aw)\s*:", re.IGNORECASE)


@dataclass
class ReplyContext:
    """Everything a threaded reply needs from the original message."""

    to: str
    cc: str
    subject: str
    in_reply_to: str
    references: str


def _reply_context(
    conn: imaplib.IMAP4_SSL, uid: str, own_address: str, reply_all: bool
) -> ReplyContext | None:
    """Fetch the original's headers by uid and resolve reply recipients.

    Headers-only ``BODY.PEEK`` fetch from a readonly INBOX select; ``None``
    when the uid no longer resolves. Keeping this inside the client means the
    threading headers never round-trip through the model.
    """
    conn.select("INBOX", readonly=True)
    _, fetched = conn.uid(
        "FETCH",
        uid,
        "(BODY.PEEK[HEADER.FIELDS "
        "(FROM REPLY-TO TO CC SUBJECT MESSAGE-ID REFERENCES IN-REPLY-TO)])",
    )
    if not fetched or not isinstance(fetched[0], tuple):
        return None
    headers = message_from_bytes(fetched[0][1])
    senders = getaddresses([headers.get("Reply-To") or headers.get("From") or ""])
    to = _require_address(senders[0][1] if senders else "")
    cc = ""
    if reply_all:
        skip = {own_address.lower(), to.lower()}
        others = [
            addr
            for _, addr in getaddresses([headers.get("To") or "", headers.get("Cc") or ""])
            if addr and addr.lower() not in skip
        ]
        cc = ", ".join(dict.fromkeys(others))
    subject = _decode(headers.get("Subject"))
    if not _REPLY_PREFIX.match(subject):
        subject = f"Re: {subject}".rstrip()
    message_id = (headers.get("Message-ID") or "").strip()
    references = (headers.get("References") or headers.get("In-Reply-To") or "").strip()
    if message_id:
        references = f"{references} {message_id}".strip()
    return ReplyContext(
        to=to, cc=cc, subject=subject, in_reply_to=message_id, references=references
    )


def _build_reply(settings: Settings, rc: ReplyContext, body: str) -> EmailMessage:
    message = _build(settings, rc.to, rc.subject, body)
    if rc.cc:
        message["Cc"] = rc.cc
    if rc.in_reply_to:
        message["In-Reply-To"] = rc.in_reply_to
    if rc.references:
        message["References"] = rc.references
    return message


def save_reply_draft(
    settings: Settings, uid: str, body: str, reply_all: bool = False
) -> str:
    """Append a threaded reply to the drafts folder. Never sends.

    The reply-path default, mirroring :func:`save_draft`.
    """
    conn = imap_connect(settings)
    try:
        rc = _reply_context(conn, uid, settings.email_address or "", reply_all)
        if rc is None:
            return f"No message with uid {uid}."
        message = _build_reply(settings, rc, body)
        stamp = imaplib.Time2Internaldate(datetime.now(UTC).timestamp())
        conn.append(settings.email_drafts_folder, r"\Draft", stamp, message.as_bytes())
    finally:
        _close(conn)
    return f"reply drafted: “{rc.subject}” to {rc.to}"


def send_reply(settings: Settings, uid: str, body: str, reply_all: bool = False) -> str:
    """Send a threaded reply. Same double gate and contract as :func:`send_message`."""
    _require_enabled(settings)
    if not settings.enable_email_send:
        # Nothing has been fetched or written at this point — same contract as
        # send_message: say so rather than claiming a draft was saved.
        raise MailDisabledError(
            "Sending is disabled, so nothing was sent or drafted. "
            "Use a draft instead, or set ENABLE_EMAIL_SEND=true to allow sending."
        )
    conn = imap_connect(settings)
    try:
        rc = _reply_context(conn, uid, settings.email_address or "", reply_all)
    finally:
        _close(conn)
    if rc is None:
        return f"No message with uid {uid}."
    message = _build_reply(settings, rc, body)
    smtp = _smtp_connect(settings)
    try:
        smtp.send_message(message)
    finally:
        with contextlib.suppress(Exception):
            smtp.quit()
    return f"reply sent: “{rc.subject}” to {rc.to}"


# --------------------------------------------------------------------------- #
# Mailbox mutations — archive / mark read / label. Each returns a one-line
# summary naming the message and its recovery path; the summaries double as
# the audit-ledger detail (assistant.mail.audit).
# --------------------------------------------------------------------------- #


def _capability_string(conn: imaplib.IMAP4_SSL) -> str:
    """CAPABILITY of the *authenticated* connection, upper-cased.

    Issued fresh rather than trusting imaplib's greeting-time cache: Gmail only
    advertises ``X-GM-EXT-1`` after login.
    """
    try:
        typ, data = conn.capability()
    except Exception:
        return ""
    if typ != "OK":
        return ""
    return b" ".join(part for part in data if part).decode("ascii", "replace").upper()


def _gmail(settings: Settings, conn: imaplib.IMAP4_SSL) -> bool:
    if settings.email_provider == "gmail":
        return True
    if settings.email_provider == "generic":
        return False
    return "X-GM-EXT-1" in _capability_string(conn)


def _supports_uidplus(conn: imaplib.IMAP4_SSL) -> bool:
    return "UIDPLUS" in _capability_string(conn)


def _quote(name: str) -> str:
    """IMAP quoted-string for a folder or label name (spaces, quotes).

    Non-ASCII names ride as UTF-8 — Gmail accepts that; servers requiring
    modified-UTF-7 are an accepted limitation of staying stdlib-only.
    """
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _summary_headers(conn: imaplib.IMAP4_SSL, uid: str) -> tuple[str, str] | None:
    """(sender, subject) for a mutation summary; ``None`` when the uid is gone."""
    _, fetched = conn.uid("FETCH", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
    if not fetched or not isinstance(fetched[0], tuple):
        return None
    headers = message_from_bytes(fetched[0][1])
    return _decode(headers.get("From")), _decode(headers.get("Subject"))


def _describe(sender: str, subject: str) -> str:
    subject = subject or "(no subject)"
    if len(subject) > 60:
        subject = subject[:59] + "…"
    return f"“{subject}” from {sender}"


def _move(conn: imaplib.IMAP4_SSL, uid: str, folder: str) -> None:
    """COPY + ``\\Deleted`` + expunge — the portable IMAP "move".

    COPY must succeed before ``\\Deleted`` is ever set, so a missing target
    folder (CREATE'd once and retried) can never destroy mail.
    """
    typ, _ = conn.uid("COPY", uid, _quote(folder))
    if typ != "OK":
        conn.create(_quote(folder))
        typ, _ = conn.uid("COPY", uid, _quote(folder))
        if typ != "OK":
            raise RuntimeError(f"could not copy message {uid} to folder {folder!r}")
    conn.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
    if _supports_uidplus(conn):
        conn.uid("EXPUNGE", uid)
    else:
        conn.expunge()


def mark_read(settings: Settings, uid: str, unread: bool = False) -> str:
    """Set (or with ``unread=True`` clear) ``\\Seen`` on one message."""
    conn = imap_connect(settings)
    try:
        conn.select("INBOX")
        named = _summary_headers(conn, uid)
        if named is None:
            return f"No message with uid {uid}."
        conn.uid("STORE", uid, "-FLAGS" if unread else "+FLAGS", r"(\Seen)")
        state = "unread" if unread else "read"
        return f"marked {state}: {_describe(*named)}"
    finally:
        _close(conn)


def archive_message(settings: Settings, uid: str) -> str:
    """Remove a message from INBOX without deleting it.

    Gmail: drop the ``\\Inbox`` label — the literal meaning of Gmail's Archive
    button; the message stays in All Mail regardless of the account's expunge
    settings. Elsewhere: move to ``email_archive_folder``.
    """
    conn = imap_connect(settings)
    try:
        conn.select("INBOX")
        named = _summary_headers(conn, uid)
        if named is None:
            return f"No message with uid {uid}."
        if _gmail(settings, conn):
            conn.uid("STORE", uid, "-X-GM-LABELS", r"(\Inbox)")
            return f"archived: {_describe(*named)} (still in All Mail)"
        folder = settings.email_archive_folder
        _move(conn, uid, folder)
        return f"archived: {_describe(*named)} (moved to {folder})"
    finally:
        _close(conn)


def set_label(settings: Settings, uid: str, label: str, remove: bool = False) -> str:
    """Apply or remove a Gmail label; on folder-based servers, adding a label
    means moving to that folder, and removal is not expressible."""
    conn = imap_connect(settings)
    try:
        conn.select("INBOX")
        named = _summary_headers(conn, uid)
        if named is None:
            return f"No message with uid {uid}."
        if _gmail(settings, conn):
            op = "-X-GM-LABELS" if remove else "+X-GM-LABELS"
            conn.uid("STORE", uid, op, f"({_quote(label)})")
            verb = "unlabeled" if remove else "labeled"
            return f"{verb} {label!r}: {_describe(*named)}"
        if remove:
            return (
                "This server has folders, not removable labels; "
                f"{_describe(*named)} can be moved to another folder instead."
            )
        _move(conn, uid, label)
        return f"moved to folder {label!r}: {_describe(*named)}"
    finally:
        _close(conn)


def _close(conn: imaplib.IMAP4_SSL) -> None:
    """Best-effort logout; a failed teardown must not mask the caller's result."""
    try:
        conn.logout()
    except Exception:
        logger.debug("imap logout failed", exc_info=True)
