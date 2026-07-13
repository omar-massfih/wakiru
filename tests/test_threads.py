"""Thread-registry tests — touch/upsert, channel filtering, and last_contact."""

from __future__ import annotations

import pytest

from assistant import threads
from assistant.calendar.context import now
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), timezone="Europe/Oslo")


def test_touch_registers_and_classifies_channels(settings) -> None:
    threads.touch(settings, "telegram:7")
    threads.touch(settings, "slack:C9:U1")
    threads.touch(settings, "cli:default")
    threads.touch(settings, "abc-123-uuid")  # HTTP/web threads have no prefix

    by_id = {t.thread_id: t.channel for t in threads.known_threads(settings)}
    assert by_id == {
        "telegram:7": "telegram",
        "slack:C9:U1": "slack",
        "cli:default": "cli",
        "abc-123-uuid": "http",
    }


def test_touch_upserts_instead_of_duplicating(settings) -> None:
    threads.touch(settings, "telegram:7")
    threads.touch(settings, "telegram:7")
    assert len(threads.known_threads(settings)) == 1


def test_known_threads_filters_by_channel(settings) -> None:
    threads.touch(settings, "telegram:7")
    threads.touch(settings, "slack:C9:U1")
    only_slack = threads.known_threads(settings, channel="slack")
    assert [t.thread_id for t in only_slack] == ["slack:C9:U1"]


def test_touch_roles_update_their_own_stamp(settings) -> None:
    threads.touch(settings, "telegram:7", user=True, assistant=False)
    info = threads.known_threads(settings)[0]
    assert info.last_user_at and not info.last_assistant_at

    threads.touch(settings, "telegram:7", user=False, assistant=True)
    info = threads.known_threads(settings)[0]
    assert info.last_user_at and info.last_assistant_at


def test_last_contact_is_latest_user_stamp(settings) -> None:
    assert threads.last_contact(settings) is None
    threads.touch(settings, "telegram:7")
    contact = threads.last_contact(settings)
    assert contact is not None
    assert abs((now(settings) - contact).total_seconds()) < 5


def test_empty_thread_id_is_ignored(settings) -> None:
    threads.touch(settings, "")
    assert threads.known_threads(settings) == []


def test_run_upkeep_touches_the_registry(settings, monkeypatch) -> None:
    from assistant import chat

    monkeypatch.setattr(chat, "update_memory", lambda *a, **k: None)
    monkeypatch.setattr(chat, "maybe_summarize", lambda *a, **k: None)
    chat.run_upkeep(None, settings, "hei", "hei tilbake", "slack:C9:U1")
    assert [t.thread_id for t in threads.known_threads(settings)] == ["slack:C9:U1"]


def test_target_threads_includes_slack_threads_in_notify_channel(settings) -> None:
    from assistant.proactive import target_threads

    slack_settings = settings.model_copy(
        update={"slack_bot_token": "xoxb-tok", "slack_notify_channel": "C9"}
    )
    threads.touch(slack_settings, "slack:C9:U1")  # saw the push (notify channel)
    threads.touch(slack_settings, "slack:C0:U1")  # a different channel — did not

    assert target_threads(slack_settings) == ["slack:C9:U1"]


def test_target_threads_without_slack_config_is_telegram_only(settings) -> None:
    from assistant.proactive import target_threads

    threads.touch(settings, "slack:C9:U1")
    assert target_threads(settings) == []  # no bot tokens configured at all
