"""Slack channel tests — signature verification, the allowlist, and the turn.

No network: ``post_message`` and the chat core are monkeypatched, so these stay
fast and offline. The HMAC verification runs for real.
"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from assistant import slack
from assistant.config import Settings

SECRET = "s3cr3t"


def _settings(**kw) -> Settings:
    base = {"slack_bot_token": "xoxb-tok", "slack_signing_secret": SECRET}
    return Settings(**{**base, **kw})


def _sign(body: bytes, timestamp: str | None = None) -> tuple[str, str]:
    timestamp = timestamp or str(int(time.time()))
    basestring = b"v0:" + timestamp.encode() + b":" + body
    sig = "v0=" + hmac.new(SECRET.encode(), basestring, hashlib.sha256).hexdigest()
    return timestamp, sig


def _event(user="U1", channel="C1", text="hello", **extra) -> dict:
    return {"event": {"type": "message", "user": user, "channel": channel, "text": text, **extra}}


@pytest.fixture
def posted(monkeypatch) -> list[tuple[str, str]]:
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(slack, "post_message", lambda tok, ch, text: sent.append((ch, text)))
    return sent


# --- signature verification -------------------------------------------------- #


def test_valid_signature_accepted() -> None:
    body = b'{"hello":"world"}'
    ts, sig = _sign(body)
    assert slack.verify_signature(SECRET, ts, body, sig) is True


def test_tampered_body_rejected() -> None:
    ts, sig = _sign(b'{"hello":"world"}')
    assert slack.verify_signature(SECRET, ts, b'{"hello":"evil"}', sig) is False


def test_wrong_secret_rejected() -> None:
    body = b"{}"
    ts, sig = _sign(body)
    assert slack.verify_signature("other-secret", ts, body, sig) is False


def test_replayed_old_timestamp_rejected() -> None:
    old = str(int(time.time()) - 60 * 10)  # 10 minutes ago
    body = b"{}"
    _, sig = _sign(body, old)
    assert slack.verify_signature(SECRET, old, body, sig) is False


def test_missing_or_malformed_inputs_rejected() -> None:
    assert slack.verify_signature("", "1", b"{}", "v0=x") is False
    assert slack.verify_signature(SECRET, "", b"{}", "v0=x") is False
    assert slack.verify_signature(SECRET, "not-a-number", b"{}", "v0=x") is False


# --- event filtering --------------------------------------------------------- #


def test_bot_message_is_ignored(posted, monkeypatch) -> None:
    settings = _settings(slack_allowed_user_ids=["U1"])
    monkeypatch.setattr(slack, "run_chat", lambda *a, **k: pytest.fail("must not answer a bot"))
    assert slack.handle_event(None, settings, _event(bot_id="B1")) is None
    assert posted == []


def test_message_subtype_is_ignored(posted, monkeypatch) -> None:
    settings = _settings(slack_allowed_user_ids=["U1"])
    monkeypatch.setattr(slack, "run_chat", lambda *a, **k: pytest.fail("must not answer an edit"))
    assert slack.handle_event(None, settings, _event(subtype="message_changed")) is None


def test_non_message_event_is_ignored(posted) -> None:
    settings = _settings(slack_allowed_user_ids=["U1"])
    assert slack.handle_event(None, settings, {"event": {"type": "reaction_added"}}) is None


# --- authorization ----------------------------------------------------------- #


def test_empty_allowlist_fails_closed(posted, monkeypatch) -> None:
    # No pairing handshake here, so an unconfigured allowlist must answer nobody.
    settings = _settings(slack_allowed_user_ids=[])
    monkeypatch.setattr(slack, "run_chat", lambda *a, **k: pytest.fail("must not chat"))
    assert slack.handle_event(None, settings, _event()) is None
    assert posted == []


def test_unauthorized_user_is_ignored(posted, monkeypatch) -> None:
    settings = _settings(slack_allowed_user_ids=["U-other"])
    monkeypatch.setattr(slack, "run_chat", lambda *a, **k: pytest.fail("must not chat"))
    assert slack.handle_event(None, settings, _event(user="U1")) is None


# --- the turn ---------------------------------------------------------------- #


def test_authorized_user_gets_reply_and_deferred_upkeep(posted, monkeypatch) -> None:
    settings = _settings(slack_allowed_user_ids=["U1"])
    monkeypatch.setattr(
        slack, "run_chat", lambda agent, text, thread, **kw: f"echo:{text} [{thread}]"
    )
    upkeep: list[tuple] = []
    monkeypatch.setattr(slack, "run_upkeep", lambda *a: upkeep.append(a))

    do_upkeep = slack.handle_event(None, settings, _event(user="U1", channel="C9", text="hei"))

    assert posted == [("C9", "echo:hei [slack:C9:U1]")]
    assert upkeep == []  # nothing runs on the reply path (Slack wants a 3s ack) …
    do_upkeep()
    assert len(upkeep) == 1  # … the route runs it in the background


def test_codex_error_posts_apology(posted, monkeypatch) -> None:
    from assistant.codex_runner import CodexError

    settings = _settings(slack_allowed_user_ids=["U1"])

    def boom(*a, **k):
        raise CodexError("nope")

    monkeypatch.setattr(slack, "run_chat", boom)
    assert slack.handle_event(None, settings, _event()) is None
    assert "hit an error" in posted[0][1]
