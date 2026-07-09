"""The email read path — rendered *on request only*.

Deliberately not wired into the agent graph: unlike the calendar, tasks, or
recall, mail is not injected into every turn. It is private correspondence and
would cost tokens on every message, so it is surfaced only when explicitly asked
for (the ``/email`` command, the ``GET /email`` endpoint).
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
