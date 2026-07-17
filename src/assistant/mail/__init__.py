"""Email — the one subsystem that talks to an external service.

Off by default (``enable_email``). Everything else in this assistant is local and
offline; mail needs real external auth (OAuth2 XOAUTH2, or an app password) and
reads private correspondence, so it is opt-in and conservative:

* **Read** — :func:`unread_summary` / :func:`list_recent` / :func:`read_message`.
  Surfaced *on request only* (the ``/email`` command, ``GET /email``), never
  injected into every turn. Uses ``BODY.PEEK``, so reading through the assistant
  never marks your mail as read.
* **Manage** — :func:`archive_message`, :func:`mark_read`, :func:`set_label`:
  deliberate, hand-recoverable mutations (archive keeps the mail; read state and
  labels have reverse verbs). Every mutation is logged to :mod:`.audit`.
* **Write** — :func:`save_draft` / :func:`save_reply_draft` are the default:
  they append to the drafts folder and send nothing. :func:`send_message` /
  :func:`send_reply` sit behind a second, independent switch
  (``enable_email_send``) and are never invoked from a background path.

Named ``mail`` rather than ``email`` so it can't shadow the stdlib module it uses.
"""

from __future__ import annotations

from . import client
from .client import (
    MailDisabledError,
    Message,
    archive_message,
    list_recent,
    mark_read,
    read_message,
    save_draft,
    save_reply_draft,
    send_message,
    send_reply,
    set_label,
)
from .context import unread_summary
from .oauth import MailAuthError

__all__ = [
    "MailAuthError",
    "MailDisabledError",
    "Message",
    "archive_message",
    "client",
    "list_recent",
    "mark_read",
    "read_message",
    "save_draft",
    "save_reply_draft",
    "send_message",
    "send_reply",
    "set_label",
    "unread_summary",
]
