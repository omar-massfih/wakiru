"""Delivery tests for :mod:`assistant.notify`.

No network: the webhook POST (``deliver_webhook``) and the Telegram transport
(``send_message`` / ``authorized_chats``) are monkeypatched, so these stay fast
and offline. They cover the fan-out and, most importantly,
``deliver_write_confirmation``'s direct-to-originating-chat routing and its
authorization guard.
"""

from __future__ import annotations

import pytest

from assistant import notify
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        telegram_bot_token="tok",
        telegram_allowed_chat_ids=[7],
        reminder_webhook_url=None,
    )


def _capture_telegram(monkeypatch) -> list[tuple[int, str]]:
    """Record every Telegram send as (chat_id, text); no network."""
    sends: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "assistant.telegram.send_message",
        lambda token, chat_id, text: sends.append((chat_id, text)),
    )
    return sends


# --- deliver_reminder fan-out --------------------------------------------- #


def test_deliver_reminder_fans_out_to_authorized_chats(settings, monkeypatch) -> None:
    sends = _capture_telegram(monkeypatch)
    delivered = notify.deliver_reminder(settings, {"message": "Dentist in 30 min"})
    assert delivered is True
    assert sends == [(7, "⏰ Dentist in 30 min")]


def test_deliver_reminder_without_any_channel_is_noop(tmp_path) -> None:
    settings = Settings(memory_dir=str(tmp_path / "m"), telegram_bot_token=None)
    assert notify.deliver_reminder(settings, {"message": "x"}) is False


def test_deliver_webhook_posts_when_url_set(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "m"), reminder_webhook_url="https://ntfy.example/topic"
    )
    posted: list[bytes] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        posted.append(request.data)
        return _Resp()

    monkeypatch.setattr("assistant.notify.urllib.request.urlopen", fake_urlopen)
    assert notify.deliver_webhook(settings, {"message": "hi", "title": "T"}) is True
    assert posted == [b"hi"]


# --- deliver_write_confirmation routing ----------------------------------- #


def test_write_confirmation_routes_to_originating_telegram_chat(settings, monkeypatch) -> None:
    sends = _capture_telegram(monkeypatch)
    ok = notify.deliver_write_confirmation(settings, "telegram:7", "created: Dentist")
    assert ok is True
    # Direct to the one originating chat — not a broadcast.
    assert sends == [(7, "created: Dentist")]


def test_write_confirmation_falls_back_to_broadcast_for_http_thread(settings, monkeypatch) -> None:
    sends = _capture_telegram(monkeypatch)
    # A non-telegram (HTTP-originated) thread id has no single chat to target, so
    # it fans out to every authorized chat via the reminder path (⏰ prefix).
    ok = notify.deliver_write_confirmation(settings, "abc-123-uuid", "created: Dentist")
    assert ok is True
    assert sends == [(7, "⏰ created: Dentist")]


def test_write_confirmation_rejects_unauthorized_chat_id(settings, monkeypatch) -> None:
    sends = _capture_telegram(monkeypatch)
    # thread_id is attacker-controllable (an HTTP caller can pass any string), so
    # a chat id that isn't actually authorized must not receive a direct send.
    # It falls back to the broadcast, which only reaches the authorized set (7).
    notify.deliver_write_confirmation(settings, "telegram:999", "secret write")
    # Reaches only the authorized chat (7), via the broadcast fallback (⏰ prefix)
    # — never a direct send to the unauthorized 999.
    assert sends == [(7, "⏰ secret write")]
    assert all(chat_id == 7 for chat_id, _ in sends)


def test_write_confirmation_without_telegram_uses_webhook_only(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "m"),
        telegram_bot_token=None,
        reminder_webhook_url="https://ntfy.example/topic",
    )
    posted: list[bytes] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "assistant.notify.urllib.request.urlopen",
        lambda request, timeout: (posted.append(request.data), _Resp())[1],
    )
    assert notify.deliver_write_confirmation(settings, "telegram:7", "hi") is True
    assert posted == [b"hi"]
