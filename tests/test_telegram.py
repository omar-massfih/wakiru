"""Telegram channel tests — chunking, authorization, the update handler, and
reminder fan-out.

No network: the Bot API transport (``_call``) and the chat core are
monkeypatched, so these stay fast and offline.
"""

from __future__ import annotations

import pytest

from assistant import notify, telegram
from assistant.codex_runner import CodexError
from assistant.config import Settings


def _settings(tmp_path, **kw) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), telegram_bot_token="tok", **kw)


def _update(chat_id: int = 7, text: str = "hello", update_id: int = 10) -> dict:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, dict]]:
    """Record every Bot API call instead of hitting the network."""
    recorded: list[tuple[str, dict]] = []

    def fake_call(token, method, payload, timeout=15):
        recorded.append((method, payload))
        return {}

    monkeypatch.setattr(telegram, "_call", fake_call)
    return recorded


def _sends(calls: list[tuple[str, dict]]) -> list[dict]:
    return [payload for method, payload in calls if method == "sendMessage"]


# --- chunking -------------------------------------------------------------- #


def test_short_reply_is_one_chunk() -> None:
    assert telegram._chunks("hi") == ["hi"]


def test_empty_reply_becomes_placeholder() -> None:
    assert telegram._chunks("   ") == ["(empty reply)"]


def test_long_reply_splits_on_newlines() -> None:
    text = "\n".join(["x" * 100] * 60)  # ~6k chars, newline every 101
    pieces = telegram._chunks(text)
    assert len(pieces) == 2
    assert all(len(p) <= telegram._MAX_MESSAGE_CHARS for p in pieces)
    assert "\n".join(pieces) == text  # nothing lost at the seam


def test_long_reply_without_newlines_splits_hard() -> None:
    pieces = telegram._chunks("a" * 9000)
    assert [len(p) for p in pieces] == [4096, 4096, 808]


# --- authorization ---------------------------------------------------------- #


def test_unauthorized_chat_with_allowlist_is_silent(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[42])
    monkeypatch.setattr(telegram, "run_chat", lambda *a: pytest.fail("must not chat"))
    telegram.handle_update(None, settings, _update(chat_id=7))
    assert calls == []


def test_first_contact_pairs_and_answers(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path)  # nobody paired or allowlisted yet
    monkeypatch.setattr(telegram, "run_chat", lambda agent, text, thread: "svar")
    monkeypatch.setattr(telegram, "run_upkeep", lambda *a: None)

    telegram.handle_update(None, settings, _update(chat_id=7, text="hei"))

    sends = _sends(calls)
    assert len(sends) == 2  # the pairing notice, then the actual answer
    assert "Paired" in sends[0]["text"]
    assert sends[1]["text"] == "svar"
    assert telegram.authorized_chats(settings) == [7]


def test_pairing_survives_restart_and_locks_out_strangers(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path)
    telegram._pair(settings, 7)

    # A fresh Settings over the same memory dir (i.e. a restart) still knows chat 7 …
    reloaded = _settings(tmp_path)
    assert telegram.authorized_chats(reloaded) == [7]

    # … and a different chat is now met with silence, not pairing.
    monkeypatch.setattr(telegram, "run_chat", lambda *a: pytest.fail("must not chat"))
    telegram.handle_update(None, reloaded, _update(chat_id=8))
    assert calls == []


def test_env_allowlist_merges_with_paired(tmp_path) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[42])
    telegram._pair(settings, 7)
    assert telegram.authorized_chats(settings) == [42, 7]


def test_non_text_update_is_ignored(tmp_path, calls) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])
    photo = {"update_id": 1, "message": {"chat": {"id": 7}, "photo": []}}
    telegram.handle_update(None, settings, photo)
    assert calls == []


# --- the chat turn ----------------------------------------------------------- #


def test_authorized_chat_gets_reply_and_upkeep(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])
    monkeypatch.setattr(
        telegram, "run_chat", lambda agent, text, thread: f"echo:{text} [{thread}]"
    )
    upkeep: list[tuple] = []
    monkeypatch.setattr(telegram, "run_upkeep", lambda *a: upkeep.append(a))

    telegram.handle_update(None, settings, _update(chat_id=7, text="hei"))

    sends = _sends(calls)
    assert len(sends) == 1
    assert sends[0]["chat_id"] == 7
    # The thread id is stable per chat, so the conversation persists.
    assert sends[0]["text"] == "echo:hei [telegram:7]"
    assert len(upkeep) == 1  # memory/summary/calendar upkeep ran after the reply


def test_codex_error_sends_apology(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])

    def boom(*args):
        raise CodexError("codex is down")

    monkeypatch.setattr(telegram, "run_chat", boom)
    monkeypatch.setattr(
        telegram, "run_upkeep", lambda *a: pytest.fail("no upkeep on a failed turn")
    )
    telegram.handle_update(None, settings, _update(chat_id=7))
    sends = _sends(calls)
    assert len(sends) == 1
    assert "error" in sends[0]["text"].lower()


# --- reminder fan-out --------------------------------------------------------- #


def test_reminders_reach_paired_chat(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    telegram._pair(settings, 7)
    sent: list[tuple[int, str]] = []
    monkeypatch.setattr(
        "assistant.telegram.send_message",
        lambda token, chat_id, text: sent.append((chat_id, text)),
    )
    delivered = notify.deliver_reminder(
        settings, {"title": "Dentist", "message": "Dentist in 30 min"}
    )
    assert delivered
    assert sent == [(7, "⏰ Dentist in 30 min")]


def test_no_telegram_config_skips_delivery(tmp_path, monkeypatch) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"))  # no token, no chats
    monkeypatch.setattr(
        "assistant.telegram.send_message",
        lambda *a: pytest.fail("must not send without telegram config"),
    )
    assert notify.deliver_telegram(settings, {"message": "x"}) is False
