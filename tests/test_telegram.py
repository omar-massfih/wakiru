"""Telegram channel tests — chunking, authorization, the update handler, and
reminder fan-out.

No network: the Bot API transport (``_call``) and the chat core are
monkeypatched, so these stay fast and offline.
"""

from __future__ import annotations

from urllib.error import HTTPError, URLError

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


def test_pairing_requires_code_roundtrip(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path)  # nobody paired or allowlisted yet
    monkeypatch.setattr(telegram, "run_chat", lambda *a, **kw: pytest.fail("must not chat"))
    telegram._pending_pairings.clear()

    # First contact does NOT pair — the chat is told to fetch the code from the
    # server log (whoever runs the server is the only one who can read it).
    telegram.handle_update(None, settings, _update(chat_id=7, text="hei"))
    assert telegram.authorized_chats(settings) == []
    assert "pairing code" in _sends(calls)[0]["text"]
    code = telegram._pending_pairings[7]

    # A wrong guess re-prompts and still doesn't pair ("not-a-code" can never
    # collide with the hex code).
    telegram.handle_update(None, settings, _update(chat_id=7, text="not-a-code"))
    assert telegram.authorized_chats(settings) == []
    assert telegram._pending_pairings[7] == code  # same code survives the typo

    # Echoing the logged code completes the handshake.
    telegram.handle_update(None, settings, _update(chat_id=7, text=f"  {code} "))
    assert telegram.authorized_chats(settings) == [7]
    assert "Paired" in _sends(calls)[-1]["text"]
    assert 7 not in telegram._pending_pairings


def test_pair_writes_atomically_with_no_leftovers(tmp_path) -> None:
    settings = _settings(tmp_path)
    telegram._pair(settings, 7)
    telegram._pair(settings, 8)
    assert telegram._paired_chats(settings) == [7, 8]
    # os.replace leaves no temp file behind for a reader to trip over.
    assert list(settings.memory_path.glob("*.tmp")) == []


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
        telegram, "run_chat", lambda agent, text, thread, **kw: f"echo:{text} [{thread}]"
    )
    upkeep: list[tuple] = []
    monkeypatch.setattr(telegram, "run_upkeep", lambda *a: upkeep.append(a))

    # The turn's upkeep comes back as a callable (the poll loop runs it in the
    # background so the next message never waits on it), not run inline.
    do_upkeep = telegram.handle_update(None, settings, _update(chat_id=7, text="hei"))

    sends = _sends(calls)
    assert len(sends) == 1
    assert sends[0]["chat_id"] == 7
    # The thread id is stable per chat, so the conversation persists.
    assert sends[0]["text"] == "echo:hei [telegram:7]"
    assert upkeep == []  # nothing ran on the reply path …
    do_upkeep()
    assert len(upkeep) == 1  # … but the deferred upkeep carries the turn


def test_codex_error_sends_apology(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])

    def boom(*args, **kwargs):
        raise CodexError("codex is down")

    monkeypatch.setattr(telegram, "run_chat", boom)
    assert telegram.handle_update(None, settings, _update(chat_id=7)) is None
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


def test_oversized_render_is_resplit_under_limit(calls) -> None:
    # Escaping expands the text ("&" -> "&amp;"), so a markdown chunk that fits
    # can render past the API limit; the sender must re-split until it fits.
    text = "\n".join(["&" * 80] * 40)  # 3.2k of markdown, ~16k once escaped
    telegram.send_message("tok", 7, text)
    sends = _sends(calls)
    assert len(sends) >= 4
    assert all(len(s["text"]) <= telegram._MAX_MESSAGE_CHARS for s in sends)
    assert all(s.get("parse_mode") == "HTML" for s in sends)
    # Nothing was lost across the splits.
    assert "".join(s["text"] for s in sends).count("&amp;") == 80 * 40


def test_network_error_on_html_send_falls_back_to_plain_text(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_call(token, method, payload, timeout=15):
        calls.append((method, payload))
        if payload.get("parse_mode") == "HTML":
            raise URLError("timed out")  # not an HTTPError: a transport failure
        return {}

    monkeypatch.setattr(telegram, "_call", fake_call)

    telegram.send_message("tok", 7, "**hello**")

    assert calls == [
        ("sendMessage", {"chat_id": 7, "text": "<b>hello</b>", "parse_mode": "HTML"}),
        ("sendMessage", {"chat_id": 7, "text": "**hello**"}),
    ]


def test_send_raises_only_when_nothing_was_delivered(monkeypatch) -> None:
    def fake_call(token, method, payload, timeout=15):
        raise URLError("network down")

    monkeypatch.setattr(telegram, "_call", fake_call)
    with pytest.raises(URLError):
        telegram.send_message("tok", 7, "hi")


def test_one_failed_chunk_does_not_kill_the_rest(monkeypatch) -> None:
    delivered: list[str] = []

    def fake_call(token, method, payload, timeout=15):
        if delivered == [] and "parse_mode" in payload:
            raise URLError("blip")  # first chunk's HTML attempt fails …
        delivered.append(payload["text"])
        return {}

    monkeypatch.setattr(telegram, "_call", fake_call)
    text = "\n".join(["x" * 100] * 60)  # two chunks
    telegram.send_message("tok", 7, text)  # must not raise
    assert len(delivered) == 2  # plain-text retry of chunk 1, then chunk 2


# --- the poll loop ----------------------------------------------------------- #


def test_poll_loop_survives_poison_updates_and_outages(tmp_path, monkeypatch) -> None:
    """One run exercises all three failure paths: a poison update must not be
    redelivered (offset still advances), a getUpdates outage must be retried,
    and a turn's upkeep must run off the reply path."""
    import asyncio
    import threading

    settings = _settings(tmp_path)
    upkeep_ran = threading.Event()
    offsets: list = []

    def fake_handle(agent, s, update):
        if update["update_id"] == 1:
            raise ValueError("poison update")
        return upkeep_ran.set  # the turn's deferred upkeep

    def fake_get_updates(token, offset):
        offsets.append(offset)
        if len(offsets) == 1:
            return [_update(update_id=1), _update(update_id=2)]
        if len(offsets) == 2:
            # Runs in a worker thread, so blocking here is fine: prove the
            # previous turn's upkeep completed in the background, then fail.
            assert upkeep_ran.wait(timeout=5), "upkeep never ran"
            raise RuntimeError("network outage")
        raise asyncio.CancelledError  # end the otherwise-infinite loop

    monkeypatch.setattr(telegram, "handle_update", fake_handle)
    monkeypatch.setattr(telegram, "get_updates", fake_get_updates)
    monkeypatch.setattr(telegram, "_RETRY_SECONDS", 0)  # no real back-off sleep

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(telegram.poll_loop(None, settings))

    # First poll starts blank; the poison update advanced the offset to 2 and
    # the good one to 3 (never redelivered); the outage retried, not died.
    assert offsets == [None, 3, 3]


# --- slash commands -------------------------------------------------------- #


def test_help_command_answers_without_model_or_upkeep(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])
    monkeypatch.setattr(
        telegram, "run_chat", lambda *a, **k: pytest.fail("commands must not call the model")
    )
    upkeep = telegram.handle_update(None, settings, _update(chat_id=7, text="/help"))
    assert upkeep is None  # commands never schedule upkeep
    sends = _sends(calls)
    assert len(sends) == 1
    assert "personal assistant" in sends[0]["text"].lower()


def test_tasks_command_lists_open_tasks(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7], timezone="Europe/Oslo")
    from assistant.tasks import store as tstore

    tstore.create_task(settings, "call plumber")
    monkeypatch.setattr(telegram, "run_chat", lambda *a, **k: pytest.fail("no model"))
    telegram.handle_update(None, settings, _update(chat_id=7, text="/tasks"))
    assert "call plumber" in _sends(calls)[0]["text"]


def test_unknown_command_shows_help(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])
    monkeypatch.setattr(telegram, "run_chat", lambda *a, **k: pytest.fail("no model"))
    telegram.handle_update(None, settings, _update(chat_id=7, text="/wat"))
    assert "Commands:" in _sends(calls)[0]["text"]


def test_slash_with_no_command_shows_help(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])
    monkeypatch.setattr(telegram, "run_chat", lambda *a, **k: pytest.fail("no model"))
    # "/" followed by only whitespace has no first word — indexing it used to
    # raise IndexError, leaving the chat with no reply at all.
    for text in ("/", "/ ", "/   "):
        telegram.handle_update(None, settings, _update(chat_id=7, text=text))
    sends = _sends(calls)
    assert len(sends) == 3
    assert all("Commands:" in s["text"] for s in sends)


def test_command_with_botname_suffix_is_stripped(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])
    monkeypatch.setattr(telegram, "run_chat", lambda *a, **k: pytest.fail("no model"))
    # In groups Telegram appends @BotName; it must still resolve to /help.
    telegram.handle_update(None, settings, _update(chat_id=7, text="/help@MyAssistantBot"))
    assert "personal assistant" in _sends(calls)[0]["text"].lower()


def test_reset_command_clears_history(tmp_path, calls, monkeypatch) -> None:
    settings = _settings(tmp_path, telegram_allowed_chat_ids=[7])
    monkeypatch.setattr(telegram, "run_chat", lambda *a, **k: pytest.fail("no model"))
    reset_calls: list[str] = []
    monkeypatch.setattr(
        telegram, "_reset_thread", lambda agent, thread_id: reset_calls.append(thread_id)
    )
    telegram.handle_update(None, settings, _update(chat_id=7, text="/reset"))
    assert reset_calls == ["telegram:7"]
    assert "forgotten" in _sends(calls)[0]["text"].lower()
