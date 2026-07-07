"""Telegram channel tests — chunking, authorization, the update handler, and
reminder fan-out.

No network: the Bot API transport (``_call``) and the chat core are
monkeypatched, so these stay fast and offline.
"""

from __future__ import annotations

import pytest
from urllib.error import HTTPError

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


# --- formatting ------------------------------------------------------------ #


def test_markdown_reply_is_sent_as_telegram_html(calls) -> None:
    telegram.send_message("tok", 7, "# Title\n\nHello **bold** and _italic_.")

    sends = _sends(calls)
    assert sends == [
        {
            "chat_id": 7,
            "text": "<b>Title</b>\n\nHello <b>bold</b> and <i>italic</i>.",
            "parse_mode": "HTML",
        }
    ]


def test_code_links_lists_and_blockquotes_render_to_supported_html(calls) -> None:
    telegram.send_message(
        "tok",
        7,
        "\n".join(
            [
                "See [site](https://example.com) and `x < y`.",
                "",
                "```python",
                "print('<ok>')",
                "```",
                "",
                "- one",
                "- two",
                "",
                "> quoted",
            ]
        ),
    )

    text = _sends(calls)[0]["text"]
    assert '<a href="https://example.com">site</a>' in text
    assert "<code>x &lt; y</code>" in text
    assert '<pre><code class="language-python">print(&#x27;&lt;ok&gt;&#x27;)</code></pre>' in text
    assert "• one\n• two" in text
    assert "<blockquote>quoted</blockquote>" in text


def test_raw_html_is_escaped(calls) -> None:
    telegram.send_message("tok", 7, "<b>not trusted</b> & raw")
    assert _sends(calls)[0]["text"] == "&lt;b&gt;not trusted&lt;/b&gt; &amp; raw"


def test_unsafe_links_render_as_plain_text(calls) -> None:
    telegram.send_message("tok", 7, "[bad](ftp://example.com)")
    assert _sends(calls)[0]["text"] == "bad"


def test_formatted_send_retries_plain_text_on_rejected_html(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_call(token, method, payload, timeout=15):
        calls.append((method, payload))
        if payload.get("parse_mode") == "HTML":
            raise RuntimeError("can't parse entities")
        return {}

    monkeypatch.setattr(telegram, "_call", fake_call)

    telegram.send_message("tok", 7, "**hello**")

    assert calls == [
        ("sendMessage", {"chat_id": 7, "text": "<b>hello</b>", "parse_mode": "HTML"}),
        ("sendMessage", {"chat_id": 7, "text": "**hello**"}),
    ]


def test_formatted_send_retries_plain_text_on_telegram_http_error(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_call(token, method, payload, timeout=15):
        calls.append((method, payload))
        if payload.get("parse_mode") == "HTML":
            raise HTTPError("url", 400, "Bad Request: can't parse entities", {}, None)
        return {}

    monkeypatch.setattr(telegram, "_call", fake_call)

    telegram.send_message("tok", 7, "**hello**")

    assert calls == [
        ("sendMessage", {"chat_id": 7, "text": "<b>hello</b>", "parse_mode": "HTML"}),
        ("sendMessage", {"chat_id": 7, "text": "**hello**"}),
    ]


def test_long_formatted_reply_still_splits_into_messages(calls) -> None:
    telegram.send_message("tok", 7, "**" + ("a" * 9000) + "**")
    sends = _sends(calls)
    assert len(sends) == 3
    assert all(send["parse_mode"] == "HTML" for send in sends)


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
