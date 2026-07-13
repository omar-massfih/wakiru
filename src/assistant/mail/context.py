"""The live email read path — a fresh IMAP fetch, rendered on request.

Used where freshness matters and a network round-trip is acceptable: the
``/email`` command and the ``GET /email`` endpoint. The per-turn context block
never calls this — it reads the cached snapshot instead
(:mod:`assistant.mail.snapshot`), so the reply path never waits on IMAP.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from . import client


def unread_summary(settings: Settings | None = None) -> str:
    """A short text listing of unread INBOX messages (sender + subject)."""
    settings = settings or get_settings()
    if not settings.enable_email:
        return "Email is off. Set ENABLE_EMAIL=true to use it."
    try:
        messages = client.list_recent(settings, unread_only=True)
    except Exception as exc:
        return f"Couldn't reach your mailbox: {exc}"
    if not messages:
        return "No unread mail."
    lines = [f"You have {len(messages)} unread message(s):", ""]
    for message in messages:
        lines.append(f"- {message.subject or '(no subject)'} — from {message.sender}")
    return "\n".join(lines)
