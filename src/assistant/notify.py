"""Outbound delivery for proactive reminders.

Deliberately tiny channels, stdlib-only (no runtime HTTP dependency):

* **Webhook** — POST the reminder to a configured URL. The default target is an
  `ntfy <https://ntfy.sh>`_ topic — the message is the body and the event title
  is the ``Title`` header — but any endpoint that accepts a plain POST works.
* **Telegram** — push the reminder to every allowed chat when the Telegram
  channel is configured, so nudges land in the same conversation you chat in.

:func:`deliver_reminder` fans out to every configured channel. Delivery is
best-effort: any failure (unset URL, network error, non-2xx) is logged and
swallowed so a push that doesn't land never breaks the reminder tick.
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


def deliver_telegram(settings: Settings, reminder: dict) -> bool:
    """Push a reminder to every authorized Telegram chat; True if any landed."""
    token = settings.telegram_bot_token
    if not token:
        return False
    # Imported here, not at module level: the calendar package (which imports
    # this module) sits on the telegram module's import path — a cycle otherwise.
    from .telegram import authorized_chats, send_message

    chats = authorized_chats(settings)
    if not chats:
        return False

    delivered = False
    for chat_id in chats:
        try:
            send_message(token, chat_id, f"⏰ {reminder.get('message', '')}")
            delivered = True
        except Exception as exc:
            logger.warning("telegram reminder delivery to %s failed: %s", chat_id, exc)
    return delivered


def deliver_reminder(settings: Settings, reminder: dict) -> bool:
    """Fan a reminder out to every configured channel; True if any delivered."""
    sent_webhook = deliver_webhook(settings, reminder)
    sent_telegram = deliver_telegram(settings, reminder)
    return sent_webhook or sent_telegram
