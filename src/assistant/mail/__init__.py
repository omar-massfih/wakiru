"""Email — the one subsystem that talks to an external service.

Off by default (``enable_email``). Everything else in this assistant is local and
offline; mail needs real external auth (OAuth2 XOAUTH2, or an app password) and
reads private correspondence, so it is opt-in and conservative:

* **Read** — :func:`unread_summary` / :func:`list_recent` / :func:`read_message`.
  Surfaced *on request only* (the ``/email`` command, ``GET /email``), never
  injected into every turn. Uses ``BODY.PEEK``, so reading through the assistant
  never marks your mail as read.
* **Write** — :func:`save_draft` is the default: it appends to the drafts folder
  and sends nothing. :func:`send_message` sits behind a second, independent
  switch (``enable_email_send``) and is never invoked from a background path.

Named ``mail`` rather than ``email`` so it can't shadow the stdlib module it uses.
"""

from __future__ import annotations

from . import client
from .client import MailDisabledError, Message, list_recent, read_message, save_draft, send_message
from .context import unread_summary
from .oauth import MailAuthError

__all__ = [
    "client",
    "Message",
    "MailAuthError",
    "MailDisabledError",
    "list_recent",
    "read_message",
    "save_draft",
    "send_message",
    "unread_summary",
]
