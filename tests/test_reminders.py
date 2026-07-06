"""Reminder tests — due computation, the dedupe ledger, pruning, and delivery.

Everything runs for real (plain SQLite + stdlib datetime); the only thing faked is
the outbound webhook POST, so these stay fast and offline.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant.calendar import context, reminders, store
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_reminders=True,
        reminder_lead_minutes=[60],
        reminder_webhook_url=None,  # no push; run_reminders still computes + records
    )


def _event_in(settings: Settings, title: str, **delta) -> store.Event:
    # Seconds precision: minute-truncation would shave up to 59s off the lead and
    # make "in 30 min" round down to 29.
    start = (context.now(settings) + timedelta(**delta)).isoformat(timespec="seconds")
    return store.create_event(settings, title=title, start=start)


def _ledger_rows(settings: Settings) -> list[dict]:
    with reminders._connect(settings) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM reminders_fired").fetchall()]


# --- due computation ------------------------------------------------------ #


def test_humanize() -> None:
    assert reminders._humanize(timedelta(minutes=30)) == "in 30 min"
    assert reminders._humanize(timedelta(minutes=60)) == "in 1 hour"
    assert reminders._humanize(timedelta(hours=2)) == "in 2 hours"
    assert reminders._humanize(timedelta(days=1)) == "in 1 day"


def test_fires_within_lead(settings) -> None:
    _event_in(settings, "Dentist", minutes=30)
    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert fired[0]["title"] == "Dentist"
    assert fired[0]["message"] == "Dentist in 30 min"
    assert fired[0]["lead_minutes"] == 60


def test_event_outside_lead_not_fired(settings) -> None:
    _event_in(settings, "Far off", hours=5)  # beyond the 60-min lead
    assert reminders.run_reminders(settings) == []


def test_past_event_not_fired(settings) -> None:
    _event_in(settings, "Missed", minutes=-10)
    assert reminders.run_reminders(settings) == []


# --- dedupe ledger -------------------------------------------------------- #


def test_dedupe_second_run_is_silent(settings) -> None:
    _event_in(settings, "Standup", minutes=15)
    assert len(reminders.run_reminders(settings)) == 1
    assert reminders.run_reminders(settings) == []  # already fired
    assert len(_ledger_rows(settings)) == 1


def test_recurring_event_fires_per_occurrence(settings) -> None:
    # A daily series whose today-occurrence is 30 min out (DTSTART a few days back).
    occ_time = context.now(settings) + timedelta(minutes=30)
    dtstart = (occ_time - timedelta(days=3)).isoformat(timespec="seconds")
    store.create_event(settings, title="Standup", start=dtstart, rrule="FREQ=DAILY")

    fired = reminders.run_reminders(settings)
    assert len(fired) == 1 and fired[0]["title"] == "Standup"
    assert reminders.run_reminders(settings) == []  # this occurrence already fired

    # Tomorrow's occurrence has a distinct start, so it is an unclaimed ledger key.
    upcoming = reminders.due_reminders(settings, current=context.now(settings) + timedelta(days=1))
    assert len(upcoming) == 1
    fired_starts = {r["event_start"] for r in _ledger_rows(settings)}
    assert upcoming[0]["start"] not in fired_starts


def test_reschedule_fires_again(settings) -> None:
    event = _event_in(settings, "Call", minutes=20)
    assert len(reminders.run_reminders(settings)) == 1

    new_start = (context.now(settings) + timedelta(minutes=45)).isoformat(timespec="minutes")
    store.update_event(settings, event.id, start=new_start)
    fired = reminders.run_reminders(settings)  # new start => new ledger key
    assert len(fired) == 1
    assert fired[0]["start"] == new_start


def test_multiple_leads_fire_only_open_window(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[1440, 60],  # a day before, and an hour before
    )
    _event_in(settings, "Flight", hours=12)  # inside the day window, outside the hour one
    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert fired[0]["lead_minutes"] == 1440


def test_ledger_prunes_old_rows(settings) -> None:
    old = (context.now(settings) - timedelta(days=40)).isoformat(timespec="seconds")
    with reminders._connect(settings) as conn:
        conn.execute(
            "INSERT INTO reminders_fired (event_id, event_start, lead_minutes, fired_at)"
            " VALUES ('stale', 'x', 60, ?)",
            (old,),
        )
    reminders.run_reminders(settings)  # prunes before firing
    assert all(r["event_id"] != "stale" for r in _ledger_rows(settings))


def test_disabled_is_noop(tmp_path) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_reminders=False)
    _event_in(settings, "Whatever", minutes=10)
    assert reminders.run_reminders(settings) == []


# --- delivery ------------------------------------------------------------- #


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_webhook_delivery(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[60],
        reminder_webhook_url="https://ntfy.example/topic",
    )
    _event_in(settings, "Dentist", minutes=30)

    calls: list[dict] = []

    def fake_urlopen(request, timeout=None):
        calls.append(
            {
                "url": request.full_url,
                "body": request.data.decode("utf-8"),
                "title": request.headers.get("Title"),
            }
        )
        return _FakeResponse()

    monkeypatch.setattr("assistant.notify.urllib.request.urlopen", fake_urlopen)

    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert len(calls) == 1
    assert calls[0]["url"] == "https://ntfy.example/topic"
    assert calls[0]["body"] == "Dentist in 30 min"
    assert calls[0]["title"] == "Dentist"


def test_no_webhook_url_skips_post(settings, monkeypatch) -> None:
    _event_in(settings, "Dentist", minutes=30)
    monkeypatch.setattr(
        "assistant.notify.urllib.request.urlopen",
        lambda *a, **k: pytest.fail("must not POST when no webhook URL is set"),
    )
    fired = reminders.run_reminders(settings)  # webhook unset in the fixture
    assert len(fired) == 1  # still computed + returned
