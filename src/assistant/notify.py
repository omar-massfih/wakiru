"""Outbound delivery for proactive reminders.

A deliberately tiny channel: POST the reminder to a configured webhook using only
the standard library (no runtime HTTP dependency). The default target is an
`ntfy <https://ntfy.sh>`_ topic — the message is the body and the event title is
the ``Title`` header — but any endpoint that accepts a plain POST works.

Delivery is best-effort: any failure (unset URL, network error, non-2xx) is logged
and swallowed so a push that doesn't land never breaks the reminder tick.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request

from .config import Settings

logger = logging.getLogger(__name__)

# Keep a failed push from stalling the tick loop.
_TIMEOUT_SECONDS = 5


def deliver_webhook(settings: Settings, reminder: dict) -> bool:
    """POST a reminder to ``reminder_webhook_url``; return whether it was sent.

    ``reminder`` is a due-reminder dict (see :func:`assistant.calendar.reminders`)
    with at least ``message`` and ``title``. Returns ``False`` (without raising)
    when no URL is configured or the request fails.
    """
    url = settings.reminder_webhook_url
    if not url:
        return False

    body = str(reminder.get("message", "")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        # ntfy reads the title from this header; harmless for generic webhooks.
        headers={"Title": str(reminder.get("title", "Reminder"))},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS):
            return True
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("reminder webhook delivery failed: %s", exc)
        return False
