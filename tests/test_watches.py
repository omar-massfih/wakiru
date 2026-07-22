"""Watch tests — the store, each kind's evaluation, wake pull, and the tools."""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant import heartbeat, threads, watches
from assistant.calendar.context import now
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_heartbeat=True,
        enable_email=True,
    )


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )


def _at(settings: Settings, **delta):
    return now(settings) + timedelta(**delta)


def _snapshot(monkeypatch, text: str) -> None:
    monkeypatch.setattr("assistant.mail.snapshot.current", lambda s: text)


# --- the store ---------------------------------------------------------------- #


def test_add_rejects_unknown_kind_and_enforces_cap(settings) -> None:
    assert watches.add(settings, "telepathy", "x") is None
    for i in range(settings.watches_max_active):
        assert watches.add(settings, "mail_from", f"sender-{i}") is not None
    assert watches.add(settings, "mail_from", "one too many") is None


def test_watch_gets_a_default_expiry_and_expires(settings) -> None:
    w = watches.add(settings, "mail_from", "Skatteetaten")
    assert w.expires_at  # mandatory expiry, defaulted
    past_expiry = now(settings) + timedelta(days=settings.watch_default_expiry_days + 1)
    assert watches.list_active(settings, past_expiry) == []
    # The expiry sweep is persistent, not per-call filtering.
    assert watches.list_active(settings) == []


def test_cancel_by_id_and_pattern_refusing_ambiguity(settings) -> None:
    a = watches.add(settings, "mail_from", "flight to Oslo")
    b = watches.add(settings, "mail_from", "flight to Bergen")
    result = watches.cancel(settings, "flight")  # ambiguous
    assert isinstance(result, list)
    assert {w.id for w in result} == {a.id, b.id}
    assert watches.cancel(settings, a.id).id == a.id
    assert watches.cancel(settings, "bergen") is not None
    assert watches.list_active(settings) == []


# --- mail_from ---------------------------------------------------------------- #


def test_mail_watch_fires_once_per_match_set(settings, monkeypatch) -> None:
    watches.add(settings, "mail_from", "Skatteetaten", note="check the tax notice")
    _snapshot(monkeypatch, "- (unread) Skatteetaten — Skatteoppgjør (today)")
    fired = watches.evaluate(settings)
    assert len(fired) == 1
    line = fired[0][1]
    assert "Skatteetaten" in line and "your note: check the tax notice" in line
    # One-shot: consumed, a second evaluation is quiet.
    assert watches.evaluate(settings) == []
    assert watches.list_active(settings) == []


def test_repeating_mail_watch_fires_on_new_matches_only(settings, monkeypatch) -> None:
    watches.add(settings, "mail_from", "newsletter", repeat=True)
    _snapshot(monkeypatch, "- newsletter #1")
    assert len(watches.evaluate(settings)) == 1
    assert watches.evaluate(settings) == []  # same match set: quiet
    _snapshot(monkeypatch, "- newsletter #2")
    assert len(watches.evaluate(settings)) == 1  # new match set: fires again
    assert len(watches.list_active(settings)) == 1  # still active


def test_mail_watch_quiet_without_email_or_match(settings, monkeypatch) -> None:
    watches.add(settings, "mail_from", "Skatteetaten")
    _snapshot(monkeypatch, "- something unrelated")
    assert watches.evaluate(settings) == []
    no_mail = Settings(
        memory_dir=settings.memory_dir, timezone="Europe/Oslo", enable_heartbeat=True
    )
    assert watches.evaluate(no_mail) == []


# --- calendar_window ---------------------------------------------------------- #


def _create_event(settings: Settings, title: str, start) -> None:
    from assistant.calendar import ops as calendar_ops

    result = calendar_ops.apply_op(
        settings,
        {"op": "create", "title": title, "start": start.isoformat(timespec="seconds")},
        "t",
        "b",
    )
    assert result


def test_calendar_watch_fires_inside_the_lead_window(settings) -> None:
    _create_event(settings, "Flight to Rome", _at(settings, minutes=20))
    watches.add(settings, "calendar_window", "flight", note="head to the train")
    fired = watches.evaluate(settings)
    assert len(fired) == 1 and "Flight to Rome" in fired[0][1]
    assert watches.evaluate(settings) == []  # one-shot


def test_calendar_watch_stays_quiet_outside_the_window(settings) -> None:
    _create_event(settings, "Flight to Rome", _at(settings, hours=5))
    watches.add(settings, "calendar_window", "flight")
    assert watches.evaluate(settings) == []


def test_calendar_watch_pulls_the_wake_to_the_window_open(settings) -> None:
    start = _at(settings, hours=5).replace(microsecond=0)
    _create_event(settings, "Flight to Rome", start)
    watches.add(settings, "calendar_window", "flight", lead_minutes=45)
    times = watches.wake_times(settings)
    assert times == [start - timedelta(minutes=45)]
    assert heartbeat.next_wake_at(settings, now(settings)) <= start - timedelta(minutes=45)


# --- silence ------------------------------------------------------------------ #


def test_silence_watch_fires_after_a_quiet_deadline(settings) -> None:
    watches.add(
        settings, "silence", "", note="nudge about the contract",
        until=_at(settings, hours=2),
    )
    assert watches.evaluate(settings) == []  # deadline not reached
    fired = watches.evaluate(settings, _at(settings, hours=3))
    assert len(fired) == 1 and "nudge about the contract" in fired[0][1]
    assert watches.evaluate(settings, _at(settings, hours=3)) == []  # one-shot


def test_silence_watch_dissolves_if_the_user_wrote(settings) -> None:
    watches.add(settings, "silence", "", until=_at(settings, hours=2))
    threads.touch(settings, "telegram:7", user=True, assistant=False)
    assert watches.evaluate(settings, _at(settings, hours=3)) == []
    # Consumed without firing: the condition can never hold now.
    assert watches.list_active(settings, _at(settings, hours=3)) == []


def test_silence_deadline_pulls_the_wake(settings) -> None:
    deadline = _at(settings, hours=2).replace(microsecond=0)
    watches.add(settings, "silence", "", until=deadline)
    assert deadline in watches.wake_times(settings)


# --- heartbeat integration ---------------------------------------------------- #


def test_fired_watch_is_scheduled_intent_in_the_report(settings, monkeypatch) -> None:
    watches.add(settings, "mail_from", "Skatteetaten", note="open it")
    _snapshot(monkeypatch, "- (unread) Skatteetaten — vedtak")
    situation = heartbeat.gather_situation(settings)
    assert situation is not None and situation.scheduled
    assert any("Skatteetaten" in hit for hit in situation.watch_hits)
    assert "Watch hit" in situation.report()


def test_active_watches_are_listed_in_the_report_info(settings) -> None:
    watches.add(settings, "mail_from", "Skatteetaten")
    situation = heartbeat.gather_situation(settings)
    assert situation is not None and not situation.scheduled
    assert "Watches you have set" in situation.report()


# --- the tools ---------------------------------------------------------------- #


def _run(settings: Settings, name: str, args: dict) -> str:
    from assistant.tools import ToolContext, execute_tool, tool_map

    spec = tool_map(settings)[name]
    return execute_tool(spec, ToolContext(settings=settings, thread_id="telegram:7"), args)


def test_watch_tool_roundtrip(settings) -> None:
    result = _run(
        settings, "watch",
        {"kind": "mail_from", "pattern": "Skatteetaten", "note": "check it"},
    )
    assert result.startswith("Watching (mail_from): Skatteetaten")
    assert "Skatteetaten" in _run(settings, "list_watches", {})
    assert _run(settings, "unwatch", {"target": "skatte"}).startswith("Stopped watching")
    assert _run(settings, "list_watches", {}) == "No active watches."


def test_watch_tool_validates_input(settings) -> None:
    assert "kind must be one of" in _run(settings, "watch", {"kind": "telepathy"})
    assert "pattern is required" in _run(settings, "watch", {"kind": "mail_from"})
    assert "needs until" in _run(settings, "watch", {"kind": "silence"})
    past = _at(settings, hours=-1).isoformat(timespec="seconds")
    assert "already in the past" in _run(
        settings, "watch", {"kind": "silence", "until": past}
    )
    assert "whole number" in _run(
        settings, "watch",
        {"kind": "calendar_window", "pattern": "x", "lead_minutes": "soonish"},
    )


def test_watch_tools_gated_by_heartbeat_flag(tmp_path) -> None:
    from assistant.tools import available_tools

    off = Settings(memory_dir=str(tmp_path / "m"))
    assert "watch" not in {t.name for t in available_tools(off)}
    on = Settings(memory_dir=str(tmp_path / "m2"), enable_heartbeat=True)
    assert {"watch", "unwatch", "list_watches"} <= {
        t.name for t in available_tools(on, mode="heartbeat")
    }
