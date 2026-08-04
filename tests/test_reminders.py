"""Reminder tests — due computation, the claim-once ledger, pruning, and wake pull.

Everything runs for real (plain SQLite + stdlib datetime). Reminders no longer
deliver anything themselves: ``surface_due`` claims the due bands and hands
them to the heartbeat's situation report (delivery lives in test_heartbeat.py
/ test_notify.py), so there is nothing to fake here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from assistant import fired_ledger
from assistant.calendar import context, reminders, store
from assistant.config import Settings
from assistant.reminder_windows import START_GRACE, next_band_change
from assistant.trips import store as trips_store


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        enable_reminders=True,
        reminder_lead_minutes=[60],
    )


def _event_in(settings: Settings, title: str, **delta) -> store.Event:
    # Seconds precision: minute-truncation would shave up to 59s off the lead and
    # make "in 30 min" round down to 29.
    start = (context.now(settings) + timedelta(**delta)).isoformat(timespec="seconds")
    return store.create_event(settings, title=title, start=start)


def _ledger_rows(settings: Settings) -> list[dict]:
    with fired_ledger.connect(reminders._LEDGER, settings) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM reminders_fired").fetchall()]


def _freeze_absolute(monkeypatch, instant: datetime) -> None:
    real_datetime = datetime

    class FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(context, "datetime", FrozenDateTime)


# --- due computation ------------------------------------------------------ #


def test_surfaces_within_lead(settings) -> None:
    _event_in(settings, "Dentist", minutes=30)
    surfaced = reminders.surface_due(settings)
    assert len(surfaced) == 1
    assert surfaced[0]["title"] == "Dentist"
    # Phrasing varies (see assistant.phrasing); the essentials must be there.
    assert "Dentist" in surfaced[0]["message"]
    assert "30 min" in surfaced[0]["message"]
    assert surfaced[0]["lead_minutes"] == 60


def test_trip_reminder_claims_absolute_window_and_uses_dst_clock(
    settings, monkeypatch
) -> None:
    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling, "New York", start="2026-03-08", end="2026-03-08",
        timezone="America/New_York",
    )
    _freeze_absolute(monkeypatch, datetime(2026, 3, 8, 6, 30, tzinfo=UTC))
    store.create_event(
        traveling, title="DST breakfast", start="2026-03-08T07:00:00+00:00"
    )

    surfaced = reminders.surface_due(traveling)
    assert len(surfaced) == 1
    assert "03:00" in surfaced[0]["message"]
    assert "30 min" in surfaced[0]["message"]


def test_timezone_less_trip_preserves_home_reminder_clock(settings, monkeypatch) -> None:
    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling, "Bergen", start="2026-03-08", end="2026-03-08", timezone=""
    )
    _freeze_absolute(monkeypatch, datetime(2026, 3, 8, 6, 30, tzinfo=UTC))
    store.create_event(
        traveling, title="Home breakfast", start="2026-03-08T07:00:00+00:00"
    )

    surfaced = reminders.surface_due(traveling)
    assert len(surfaced) == 1
    assert "08:00" in surfaced[0]["message"]


def test_event_outside_lead_not_surfaced(settings) -> None:
    _event_in(settings, "Far off", hours=5)  # beyond the 60-min lead
    assert reminders.surface_due(settings) == []


def test_past_event_not_surfaced(settings) -> None:
    _event_in(settings, "Missed", minutes=-10)  # beyond START_GRACE
    assert reminders.surface_due(settings) == []


def test_at_start_nudge_surfaces_once(settings) -> None:
    # The moment the user asked to be reminded at gets its own band: an event
    # that just started (the wake lands a little late) surfaces the at-start
    # band, keyed as lead 0, exactly once.
    _event_in(settings, "Standup", minutes=-1)
    surfaced = reminders.surface_due(settings)
    assert len(surfaced) == 1
    assert surfaced[0]["lead_minutes"] == 0
    assert "now" in surfaced[0]["message"]
    assert reminders.surface_due(settings) == []  # claimed; later wakes silent


# --- dedupe ledger -------------------------------------------------------- #


def test_dedupe_second_run_is_silent(settings) -> None:
    _event_in(settings, "Standup", minutes=15)
    assert len(reminders.surface_due(settings)) == 1
    assert reminders.surface_due(settings) == []  # already claimed
    assert len(_ledger_rows(settings)) == 1


def test_recurring_event_surfaces_per_occurrence(settings) -> None:
    # A daily series whose today-occurrence is 30 min out (DTSTART a few days back).
    occ_time = context.now(settings) + timedelta(minutes=30)
    dtstart = (occ_time - timedelta(days=3)).isoformat(timespec="seconds")
    store.create_event(settings, title="Standup", start=dtstart, rrule="FREQ=DAILY")

    surfaced = reminders.surface_due(settings)
    assert len(surfaced) == 1 and surfaced[0]["title"] == "Standup"
    assert reminders.surface_due(settings) == []  # this occurrence already claimed

    # Tomorrow's occurrence has a distinct start, so it is an unclaimed ledger key.
    upcoming = reminders.due_reminders(settings, current=context.now(settings) + timedelta(days=1))
    assert len(upcoming) == 1
    fired_starts = {r["event_start"] for r in _ledger_rows(settings)}
    assert upcoming[0]["start"] not in fired_starts


def test_reschedule_surfaces_again(settings) -> None:
    event = _event_in(settings, "Call", minutes=20)
    assert len(reminders.surface_due(settings)) == 1

    new_start = (context.now(settings) + timedelta(minutes=45)).isoformat(timespec="minutes")
    store.update_event(settings, event.id, start=new_start)
    surfaced = reminders.surface_due(settings)  # new start => new ledger key
    assert len(surfaced) == 1
    assert surfaced[0]["start"] == new_start


def test_multiple_leads_surface_only_open_window(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        reminder_lead_minutes=[1440, 60],  # a day before, and an hour before
    )
    _event_in(settings, "Flight", hours=12)  # inside the day window, outside the hour one
    surfaced = reminders.surface_due(settings)
    assert len(surfaced) == 1
    assert surfaced[0]["lead_minutes"] == 1440


def test_ledger_prunes_old_rows(settings) -> None:
    old = (context.now(settings) - timedelta(days=40)).isoformat(timespec="seconds")
    with fired_ledger.connect(reminders._LEDGER, settings) as conn:
        conn.execute(
            "INSERT INTO reminders_fired (event_id, event_start, lead_minutes, fired_at)"
            " VALUES ('stale', 'x', 60, ?)",
            (old,),
        )
    reminders.surface_due(settings)  # prunes before claiming
    assert all(r["event_id"] != "stale" for r in _ledger_rows(settings))


def test_disabled_is_noop(tmp_path) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_reminders=False)
    _event_in(settings, "Whatever", minutes=10)
    assert reminders.surface_due(settings) == []


def test_event_inside_several_lead_windows_surfaces_once(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        reminder_lead_minutes=[1440, 60],
    )
    # Booked half an hour ahead: inside BOTH windows -> one reminder, not two
    # identical "in 30 min" lines.
    _event_in(settings, "Flight", minutes=30)
    surfaced = reminders.surface_due(settings)
    assert len(surfaced) == 1
    assert surfaced[0]["lead_minutes"] == 60  # reported at the tightest lead
    # Both leads are claimed together, so no later wake can surface a duplicate.
    assert {r["lead_minutes"] for r in _ledger_rows(settings)} == {60, 1440}
    assert reminders.surface_due(settings) == []


# --- repeat mode ---------------------------------------------------------- #


def test_repeat_surfaces_each_band_until_start(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=60)).isoformat(timespec="seconds")
    store.create_event(settings, title="Dentist", start=start)

    messages: list[str] = []
    # Walk wall-clock from 60 min out to the start in 15-min steps.
    for step in range(0, 61, 15):
        messages += [
            r["message"]
            for r in reminders.surface_due(settings, current=base + timedelta(minutes=step))
        ]

    # One nudge per 15-min band: 60, 45, 30, 15, 0 min out.
    assert len(messages) == 5
    assert all("Dentist" in m for m in messages)
    for m, countdown in zip(messages, ["1 hour", "45 min", "30 min", "15 min", "now"], strict=True):
        assert countdown in m
    assert {r["lead_minutes"] for r in _ledger_rows(settings)} == {60, 45, 30, 15, 0}


def test_repeat_same_band_is_idempotent(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=40)).isoformat(timespec="seconds")
    store.create_event(settings, title="Call", start=start)

    assert len(reminders.surface_due(settings, current=base)) == 1  # remaining 40 -> slot 30
    assert reminders.surface_due(settings, current=base) == []  # same band, already claimed


def test_repeat_silent_after_start(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=10)).isoformat(timespec="seconds")
    store.create_event(settings, title="Gone", start=start)

    # 15 min past start -> nothing.
    assert reminders.surface_due(settings, current=base + timedelta(minutes=25)) == []


def test_repeat_at_start_band_surfaces_once(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=10)).isoformat(timespec="seconds")
    store.create_event(settings, title="Kickoff", start=start)

    # The wake lands 40s after the start (loop jitter): one "starting now".
    surfaced = reminders.surface_due(settings, current=base + timedelta(minutes=10, seconds=40))
    assert len(surfaced) == 1
    assert "now" in surfaced[0]["message"]
    # Next wake, still inside the grace window: the band is claimed, no repeat.
    assert reminders.surface_due(settings, current=base + timedelta(minutes=11, seconds=40)) == []


def test_repeat_skip_occurrence_stops_remaining_nudges(tmp_path) -> None:
    # Regression for the "I'm sick today" incident: after "Exercise in 30 min"
    # surfaced, skipping today's occurrence (what the agent does when the user
    # declines) must silence the rest of the countdown — the ledger only
    # dedupes, it must not keep the schedule alive past the EXDATE.
    from assistant.calendar import ops

    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,  # uniform-lead semantics under test
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    occ = base + timedelta(minutes=30)
    dtstart = (occ - timedelta(days=3)).isoformat(timespec="seconds")
    event = store.create_event(settings, title="Exercise", start=dtstart, rrule="FREQ=DAILY")

    surfaced = reminders.surface_due(settings, current=base)
    assert len(surfaced) == 1
    assert "Exercise" in surfaced[0]["message"] and "30 min" in surfaced[0]["message"]

    assert ops.apply_op(
        settings, {"op": "skip", "id": event.id, "occurrence": occ.isoformat()}
    ) is not None
    for step in (14, 25, 29):  # the "in 14 min" nudge and every later band
        current = base + timedelta(minutes=30 - step)
        assert reminders.surface_due(settings, current=current) == []

    # Tomorrow's occurrence is untouched and nudges normally.
    refired = reminders.surface_due(settings, current=base + timedelta(days=1))
    assert len(refired) == 1
    assert "Exercise" in refired[0]["message"] and "30 min" in refired[0]["message"]


def test_ledger_prune_compares_instants_not_strings(settings) -> None:
    # A fresh row stamped under another UTC offset sorts lexically before the
    # cutoff string; pruning must compare instants and keep it.

    fresh_other_offset = (context.now(settings) - timedelta(days=1)).astimezone(UTC)
    with fired_ledger.connect(reminders._LEDGER, settings) as conn:
        conn.execute(
            "INSERT INTO reminders_fired (event_id, event_start, lead_minutes, fired_at)"
            " VALUES ('fresh', 'x', 60, ?)",
            (fresh_other_offset.isoformat(timespec="seconds"),),
        )
    reminders.surface_due(settings)  # prunes before claiming
    assert any(r["event_id"] == "fresh" for r in _ledger_rows(settings))


# --- importance tiers ------------------------------------------------------ #


@pytest.fixture
def tiered_settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=True,
        reminder_lead_minutes=[15],
        reminder_lead_minutes_critical=[2880, 1440, 180, 60, 15],
    )


def _stub_tiers(monkeypatch, verdicts: dict[str, str]) -> None:
    from assistant.calendar import importance

    monkeypatch.setattr(
        importance, "_classify_llm", lambda settings, events: dict(verdicts)
    )


def test_critical_event_surfaces_days_ahead(tiered_settings, monkeypatch) -> None:
    event = _event_in(tiered_settings, "Legetime hos Dr. Berg", hours=36)
    _stub_tiers(monkeypatch, {event.id: "critical"})
    surfaced = reminders.surface_due(tiered_settings)
    assert len(surfaced) == 1
    assert surfaced[0]["tier"] == "critical"
    assert surfaced[0]["lead_minutes"] == 2880  # the 2-day window is open
    assert "2 days" in surfaced[0]["message"]
    assert reminders.surface_due(tiered_settings) == []  # ledger holds


def test_normal_event_silent_days_ahead(tiered_settings, monkeypatch) -> None:
    event = _event_in(tiered_settings, "Coffee with Anna", hours=36)
    _stub_tiers(monkeypatch, {event.id: "normal"})
    assert reminders.surface_due(tiered_settings) == []


def test_normal_event_surfaces_inside_short_lead(tiered_settings, monkeypatch) -> None:
    event = _event_in(tiered_settings, "Coffee with Anna", minutes=10)
    _stub_tiers(monkeypatch, {event.id: "normal"})
    surfaced = reminders.surface_due(tiered_settings)
    assert len(surfaced) == 1
    assert surfaced[0]["tier"] == "normal"
    assert surfaced[0]["lead_minutes"] == 15


def test_flag_off_ignores_criticality(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_importance_enabled=False,
        reminder_lead_minutes=[15],
    )
    _event_in(settings, "Legetime", hours=36)  # would be critical if classified
    assert reminders.surface_due(settings) == []


def test_classifier_crash_degrades_to_normal_leads(tiered_settings, monkeypatch) -> None:
    from assistant.calendar import importance

    def boom(settings, events):
        raise RuntimeError("tiers exploded")

    monkeypatch.setattr(importance, "tiers_for", boom)
    _event_in(tiered_settings, "Legetime", minutes=10)  # inside the normal lead
    surfaced = reminders.surface_due(tiered_settings)
    assert len(surfaced) == 1  # reminders never blocked by classification
    assert surfaced[0]["tier"] == "normal"


# --- wake scheduling (next_band_change / next_due_at) ---------------------- #


def test_next_band_change_lead_mode() -> None:
    # 90 min out with a 60-min lead: the band opens in 30 min.
    change = next_band_change(
        timedelta(minutes=90), [60], 0, repeat_floor=timedelta(0)
    )
    assert change == timedelta(minutes=30)
    # Already inside the lead: the only future opening is the at-start nudge.
    change = next_band_change(
        timedelta(minutes=30), [60], 0, repeat_floor=timedelta(0)
    )
    assert change == timedelta(minutes=31)  # start + the 1-min nudge
    # Started and past grace: nothing left.
    assert (
        next_band_change(timedelta(minutes=-10), [60], 0, repeat_floor=timedelta(0))
        is None
    )


def test_next_band_change_picks_soonest_lead() -> None:
    # 20h out: the 1440-min band is already open (not a future opening); the
    # next opening is the 60-min band, 19h from now.
    change = next_band_change(
        timedelta(hours=20), [1440, 60], 0, repeat_floor=timedelta(0)
    )
    assert change == timedelta(hours=19)


def test_next_band_change_repeat_mode() -> None:
    # 40 min out, 60-min window, 15-min repeat: currently in slot 30; the next
    # band (slot 15... boundary at remaining=30) opens in 10 min + nudge.
    change = next_band_change(
        timedelta(minutes=40), [60], 15, repeat_floor=-START_GRACE
    )
    assert change == timedelta(minutes=11)
    # Ahead of the window: the opening is the first candidate.
    change = next_band_change(
        timedelta(minutes=90), [60], 15, repeat_floor=-START_GRACE
    )
    assert change == timedelta(minutes=30)
    # Past the floor: nothing left.
    assert (
        next_band_change(timedelta(minutes=-20), [60], 15, repeat_floor=-START_GRACE)
        is None
    )


def test_calendar_next_due_at_pulls_to_window_opening(settings) -> None:
    current = context.now(settings)
    _event_in(settings, "Dentist", minutes=90)  # 60-min lead opens in ~30 min
    openings = reminders.next_due_at(settings, current, current + timedelta(hours=2))
    assert len(openings) == 1
    lateness = openings[0] - (current + timedelta(minutes=30))
    assert abs(lateness) < timedelta(seconds=5)


def test_calendar_next_due_at_respects_horizon(settings) -> None:
    current = context.now(settings)
    _event_in(settings, "Dentist", minutes=90)
    # The opening (~30 min out) is beyond a 10-min horizon.
    assert reminders.next_due_at(settings, current, current + timedelta(minutes=10)) == []


def test_calendar_next_due_at_uses_cached_tier_only(tiered_settings, monkeypatch) -> None:
    from assistant.calendar import importance

    event = _event_in(tiered_settings, "Legetime", hours=36)
    monkeypatch.setattr(
        importance,
        "_classify_llm",
        lambda *a, **k: pytest.fail("next_due_at must never classify"),
    )
    current = context.now(tiered_settings)
    # Unclassified: treated as normal (15-min lead), far outside the horizon.
    assert (
        reminders.next_due_at(tiered_settings, current, current + timedelta(hours=13))
        == []
    )
    # Once cached as critical, the 1440-min band opening (~12h out) pulls a wake.
    with importance._connect(tiered_settings) as conn:
        conn.execute(
            "INSERT INTO event_importance (event_id, title_hash, tier, source, updated)"
            " VALUES (?, ?, 'critical', 'llm', ?)",
            (
                event.id,
                importance._title_hash("Legetime"),
                current.isoformat(timespec="seconds"),
            ),
        )
    openings = reminders.next_due_at(tiered_settings, current, current + timedelta(hours=13))
    assert len(openings) == 1
    lateness = openings[0] - (current + timedelta(hours=12))
    assert abs(lateness) < timedelta(seconds=5)
