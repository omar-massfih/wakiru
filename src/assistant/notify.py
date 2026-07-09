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

import base64
import logging
import urllib.error
import urllib.request

from .config import Settings

logger = logging.getLogger(__name__)

# Keep a failed push from stalling the tick loop.
_TIMEOUT_SECONDS = 5


def _header_value(text: str) -> str:
    """A value urllib can put in an HTTP header: Latin-1 as-is, else RFC 2047.

    urllib encodes header values as Latin-1 and raises ``UnicodeEncodeError`` on
    anything outside it — an emoji in an event title must not crash delivery.
    ntfy decodes RFC 2047 encoded words in the ``Title`` header.
    """
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return f"=?utf-8?B?{encoded}?="


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
        headers={"Title": _header_value(str(reminder.get("title", "Reminder")))},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS):
            return True
    # ValueError covers what urllib raises on a header/URL it cannot encode.
    except (urllib.error.URLError, OSError, ValueError) as exc:
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


def deliver_slack(settings: Settings, reminder: dict, channel: str | None = None) -> bool:
    """Push a reminder to a Slack channel (``slack_notify_channel`` by default)."""
    token = settings.slack_bot_token
    channel = channel or settings.slack_notify_channel
    if not (token and channel):
        return False
    # Imported here, not at module level: see deliver_telegram's note on the cycle.
    from .slack import post_message

    try:
        post_message(token, channel, f"⏰ {reminder.get('message', '')}")
        return True
    except Exception as exc:
        logger.warning("slack reminder delivery failed: %s", exc)
        return False


def deliver_reminder(settings: Settings, reminder: dict) -> bool:
    """Fan a reminder out to every configured channel; True if any delivered."""
    sent_webhook = deliver_webhook(settings, reminder)
    sent_telegram = deliver_telegram(settings, reminder)
    sent_slack = deliver_slack(settings, reminder)
    return sent_webhook or sent_telegram or sent_slack


def deliver_write_confirmation(settings: Settings, thread_id: str, message: str) -> bool:
    """Push a calendar-write confirmation (with its undo hint) back to ``thread_id``.

    Always POSTs the webhook (a distinct device channel, as with reminders). For a
    Telegram thread (``"telegram:<chat_id>"``) whose chat is authorized, the
    message goes directly to that one chat rather than fanning out to every
    authorized chat — a write belongs to the conversation that triggered it, not
    every paired chat. A Slack thread (``"slack:<channel>:<user>"``) is routed to
    its channel the same way. Falls back to the broad fan-out for a non-Telegram
    (HTTP-originated) thread or an unrecognized/unauthorized chat id.
    """
    sent_webhook = deliver_webhook(settings, {"title": "Calendar updated", "message": message})

    if thread_id.startswith("slack:"):
        # "slack:<channel>:<user>" — answer in the channel that triggered the write,
        # but only for a user we actually answer (thread_id is not a trust boundary).
        from .slack import authorized_users, post_message

        _, _, rest = thread_id.partition(":")
        channel, _, user = rest.partition(":")
        token = settings.slack_bot_token
        if token and channel and user in authorized_users(settings):
            try:
                post_message(token, channel, message)
                return True
            except Exception as exc:
                logger.warning("write-confirmation delivery to slack %s failed: %s", channel, exc)
        return sent_webhook

    token = settings.telegram_bot_token
    if not token:
        return sent_webhook
    # Imported here, not at module level: see deliver_telegram's note on the cycle.
    from .telegram import authorized_chats, send_message

    if thread_id.startswith("telegram:"):
        try:
            chat_id = int(thread_id.removeprefix("telegram:"))
        except ValueError:
            chat_id = None
        # Only send directly to a chat that is actually authorized — thread_id
        # is not itself a trust boundary (an HTTP caller could pass any string).
        if chat_id is not None and chat_id in authorized_chats(settings):
            try:
                send_message(token, chat_id, message)
                return True
            except Exception as exc:
                logger.warning("write-confirmation delivery to %s failed: %s", chat_id, exc)
                return sent_webhook

    sent_telegram = deliver_telegram(settings, {"message": message})
    return sent_webhook or sent_telegram
