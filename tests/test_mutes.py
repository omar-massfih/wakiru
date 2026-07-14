"""Reminder-mute tests — the shared quiet switch between agent and tickers.

Everything runs for real (plain SQLite + stdlib datetime); only the clock is
monkeypatched, the same discipline as test_reminders.py. The scenario pinned
throughout: the user declines a nudge in chat and the remaining nudges hold.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant import briefing, mutes
from assistant.calendar import context, store
from assistant.calendar import reminders as calendar_reminders
from assistant.config import Settings
from assistant.tasks import reminders as task_reminders
from assistant.tasks import store as tasks_store


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_reminders=True,
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
        reminder_webhook_url=None,
    )


@pytest.fixture(autouse=True)
def _compose_fallback(monkeypatch) -> None:
    """Stand-in composer: behaves like a failed model (returns the fallback)."""
    monkeypatch.setattr(
        "assistant.compose.compose_push", lambda s, **kw: kw["fallback"]
    )


def _event_in(settings: Settings, title: str, **delta) -> store.Event:
    start = (context.now(settings) + timedelta(**delta)).isoformat(timespec="seconds")
    return store.create_event(settings, title=title, start=start)


# --- store ------------------------------------------------------------------ #


def test_mute_upsert_replaces_and_expires(settings) -> None:
    # Seconds precision: until is stored (and read back) truncated to seconds.
    current = context.now(settings).replace(microsecond=0)
    mutes.set_mute(settings, "event", "e1", current + timedelta(hours=1), current=current)
    mutes.set_mute(settings, "event", "e1", current + timedelta(hours=2), current=current)
    active = mutes.active_mutes(settings, current)
    assert set(active) == {("event", "e1")}  # one row, latest until wins
    assert active[("event", "e1")] == current + timedelta(hours=2)
    # Past its expiry the mute is simply inactive.
    assert mutes.active_mutes(settings, current + timedelta(hours=3)) == {}


def test_expired_mutes_pruned_on_write(settings) -> None:
    current = context.now(settings)
    mutes.set_mute(settings, "event", "old", current + timedelta(minutes=5), current=current)
    later = current + timedelta(hours=1)
    mutes.set_mute(settings, "task", "t1", later + timedelta(hours=1), current=later)
    with mutes._connect(settings) as conn:
        rows = conn.execute("SELECT scope, target_id FROM reminder_mutes").fetchall()
    assert [(r["scope"], r["target_id"]) for r in rows] == [("task", "t1")]


def test_clear_mute(settings) -> None:
    current = context.now(settings)
    mutes.set_mute(settings, "event", "e1", current + timedelta(hours=1), current=current)
    assert mutes.clear_mute(settings, "event", "e1") is True
    assert mutes.clear_mute(settings, "event", "e1") is False  # already gone
    assert mutes.active_mutes(settings, current) == {}


# --- calendar reminders hold ------------------------------------------------ #


def test_muted_event_fires_nothing_until_mute_expires(settings, monkeypatch) -> None:
    base = context.now(settings).replace(second=0, microsecond=0)
    # Built from the frozen clock, not _event_in's real now: truncating base to
    # the whole minute would otherwise leave the event just past the 60-min lead.
    start = (base + timedelta(minutes=60)).isoformat(timespec="seconds")
    event = store.create_event(settings, title="Exercise", start=start)

    monkeypatch.setattr(calendar_reminders, "now", lambda s: base)
    assert len(calendar_reminders.run_reminders(settings)) == 1  # first nudge lands

    # "I'm sick today" -> the agent mutes the event for half an hour.
    mutes.set_mute(settings, "event", event.id, base + timedelta(minutes=35), current=base)
    monkeypatch.setattr(calendar_reminders, "now", lambda s: base + timedelta(minutes=15))
    assert calendar_reminders.run_reminders(settings) == []  # band held, not claimed
    monkeypatch.setattr(calendar_reminders, "now", lambda s: base + timedelta(minutes=30))
    assert calendar_reminders.run_reminders(settings) == []

    # Mute expired mid-window: the remaining band fires normally.
    monkeypatch.setattr(calendar_reminders, "now", lambda s: base + timedelta(minutes=45))
    fired = calendar_reminders.run_reminders(settings)
    assert len(fired) == 1
    assert "Exercise" in fired[0]["message"] and "15 min" in fired[0]["message"]


def test_mute_holds_only_its_event(settings, monkeypatch) -> None:
    base = context.now(settings).replace(second=0, microsecond=0)
    muted = _event_in(settings, "Exercise", minutes=30)
    _event_in(settings, "Dentist", minutes=30)
    mutes.set_mute(settings, "event", muted.id, base + timedelta(hours=1), current=base)

    monkeypatch.setattr(calendar_reminders, "now", lambda s: base)
    fired = calendar_reminders.run_reminders(settings)
    assert [r["title"] for r in fired] == ["Dentist"]


def test_unmute_resumes_nudges(settings, monkeypatch) -> None:
    base = context.now(settings).replace(second=0, microsecond=0)
    event = _event_in(settings, "Exercise", minutes=30)
    mutes.set_mute(settings, "event", event.id, base + timedelta(hours=1), current=base)

    monkeypatch.setattr(calendar_reminders, "now", lambda s: base)
    assert calendar_reminders.run_reminders(settings) == []
    mutes.clear_mute(settings, "event", event.id)
    assert len(calendar_reminders.run_reminders(settings)) == 1


# --- task reminders hold ------------------------------------------------------ #


def test_muted_task_stops_nagging(settings, monkeypatch) -> None:
    base = context.now(settings).replace(second=0, microsecond=0)
    due = (base + timedelta(minutes=30)).isoformat(timespec="seconds")
    task = tasks_store.create_task(settings, "Pay bill", due=due)

    monkeypatch.setattr(task_reminders, "now", lambda s: base)
    assert len(task_reminders.run_task_reminders(settings)) == 1

    mutes.set_mute(settings, "task", task.id, base + timedelta(hours=2), current=base)
    monkeypatch.setattr(task_reminders, "now", lambda s: base + timedelta(minutes=15))
    assert task_reminders.run_task_reminders(settings) == []


# --- the all scope holds everything ------------------------------------------- #


def test_all_mute_holds_calendar_tasks_and_briefing(settings, monkeypatch) -> None:
    base = context.now(settings).replace(second=0, microsecond=0)
    _event_in(settings, "Exercise", minutes=30)
    due = (base + timedelta(minutes=30)).isoformat(timespec="seconds")
    tasks_store.create_task(settings, "Pay bill", due=due)
    mutes.set_mute(settings, "all", "", base + timedelta(hours=8), current=base)

    monkeypatch.setattr(calendar_reminders, "now", lambda s: base)
    monkeypatch.setattr(task_reminders, "now", lambda s: base)
    assert calendar_reminders.run_reminders(settings) == []
    assert task_reminders.run_task_reminders(settings) == []

    settings.enable_briefing = True
    monkeypatch.setattr(briefing, "now", lambda s: base.replace(hour=8, minute=0))
    monkeypatch.setattr(
        briefing, "deliver_reminder", lambda s, r: pytest.fail("must not deliver while muted")
    )
    assert briefing.run_briefing(settings) == {"sent": False, "reason": "muted"}
