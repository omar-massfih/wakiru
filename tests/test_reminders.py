"""Reminder tests — due computation, the dedupe ledger, pruning, and delivery.

Everything runs for real (plain SQLite + stdlib datetime); faked are the
outbound webhook POST and the model composition (stubbed to its deterministic
fallback — compose_push's own behavior lives in test_compose.py), so these
stay fast and offline.
"""

from __future__ import annotations

from datetime import UTC, timedelta

import pytest

from assistant import fired_ledger
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


@pytest.fixture(autouse=True)
def _compose_fallback(monkeypatch) -> None:
    """Stand-in composer: behaves like a failed model (returns the fallback)."""
    monkeypatch.setattr(
        "assistant.compose.compose_push", lambda s, **kw: kw["fallback"]
    )


def _event_in(settings: Settings, title: str, **delta) -> store.Event:
    # Seconds precision: minute-truncation would shave up to 59s off the lead and
    # make "in 30 min" round down to 29.
    start = (context.now(settings) + timedelta(**delta)).isoformat(timespec="seconds")
    return store.create_event(settings, title=title, start=start)


def _ledger_rows(settings: Settings) -> list[dict]:
    with fired_ledger.connect(reminders._LEDGER, settings) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM reminders_fired").fetchall()]


# --- due computation ------------------------------------------------------ #


def test_fires_within_lead(settings) -> None:
    _event_in(settings, "Dentist", minutes=30)
    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert fired[0]["title"] == "Dentist"
    # Phrasing varies (see assistant.phrasing); the essentials must be there.
    assert "Dentist" in fired[0]["message"]
    assert "30 min" in fired[0]["message"]
    assert fired[0]["lead_minutes"] == 60


def test_event_outside_lead_not_fired(settings) -> None:
    _event_in(settings, "Far off", hours=5)  # beyond the 60-min lead
    assert reminders.run_reminders(settings) == []


def test_past_event_not_fired(settings) -> None:
    _event_in(settings, "Missed", minutes=-10)  # beyond START_GRACE
    assert reminders.run_reminders(settings) == []


def test_at_start_nudge_fires_once(settings) -> None:
    # The moment the user asked to be reminded at gets its own push: an event
    # that just started (the ticker lands a little late) fires the at-start
    # band, keyed as lead 0, exactly once.
    _event_in(settings, "Standup", minutes=-1)
    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert fired[0]["lead_minutes"] == 0
    assert "now" in fired[0]["message"]
    assert reminders.run_reminders(settings) == []  # claimed; later ticks silent


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
    with fired_ledger.connect(reminders._LEDGER, settings) as conn:
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
    # One batched push; with composition falling back, its body is the template.
    assert calls[0]["body"] == fired[0]["message"]
    assert "Dentist" in calls[0]["body"] and "30 min" in calls[0]["body"]
    assert calls[0]["title"] == "Reminder"


def test_no_webhook_url_skips_post(settings, monkeypatch) -> None:
    _event_in(settings, "Dentist", minutes=30)
    monkeypatch.setattr(
        "assistant.notify.urllib.request.urlopen",
        lambda *a, **k: pytest.fail("must not POST when no webhook URL is set"),
    )
    fired = reminders.run_reminders(settings)  # webhook unset in the fixture
    assert len(fired) == 1  # still computed + returned


def test_non_latin1_title_still_delivers(tmp_path, monkeypatch) -> None:
    # urllib encodes headers as Latin-1; an emoji title used to raise inside the
    # ledger transaction and wedge every reminder until the event passed. The
    # push title is a fixed "Reminder" now, but the emoji still rides in the
    # body and the claim must survive delivery.
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[60],
        reminder_webhook_url="https://ntfy.example/topic",
    )
    _event_in(settings, "Trening 💪", minutes=30)

    calls: list[dict] = []

    def fake_urlopen(request, timeout=None):
        title = request.headers.get("Title")
        title.encode("latin-1")  # what http.client does; must not raise
        calls.append({"title": title, "body": request.data.decode("utf-8")})
        return _FakeResponse()

    monkeypatch.setattr("assistant.notify.urllib.request.urlopen", fake_urlopen)

    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert len(_ledger_rows(settings)) == 1
    assert len(calls) == 1
    assert "Trening 💪" in calls[0]["body"]


def test_latin1_title_passes_through_unencoded(tmp_path, monkeypatch) -> None:
    from assistant.notify import _header_value

    assert _header_value("Møte på jobb") == "Møte på jobb"  # Latin-1-safe as-is


def test_delivery_crash_keeps_claim(settings, monkeypatch) -> None:
    # Delivery runs outside the ledger transaction and guarded: a push that
    # blows up must not roll back the claims (which would make the tick
    # re-fail forever).
    _event_in(settings, "First", minutes=10)
    _event_in(settings, "Second", minutes=20)

    def boom(settings_, reminder):
        raise UnicodeEncodeError("latin-1", "x", 0, 1, "boom")

    monkeypatch.setattr(reminders, "deliver_reminder", boom)

    fired = reminders.run_reminders(settings)
    assert {r["title"] for r in fired} == {"First", "Second"}
    assert len(_ledger_rows(settings)) == 2  # both claims survived the crash
    assert reminders.run_reminders(settings) == []  # and are not re-fired


def test_batch_composes_one_push_covering_all_due(settings, monkeypatch) -> None:
    # Several due reminders become ONE composed push; the model gets the
    # template lines as facts and the joined templates as fallback.
    _event_in(settings, "First", minutes=10)
    _event_in(settings, "Second", minutes=20)

    composed: dict = {}

    def fake_compose(s, **kwargs):
        composed.update(kwargs)
        return "Snart: First (10 min) og Second (20 min)."

    monkeypatch.setattr("assistant.compose.compose_push", fake_compose)
    pushes: list[dict] = []
    monkeypatch.setattr(reminders, "deliver_reminder", lambda s, r: pushes.append(r) or True)

    fired = reminders.run_reminders(settings)
    assert {r["title"] for r in fired} == {"First", "Second"}
    assert len(pushes) == 1
    assert pushes[0]["message"] == "Snart: First (10 min) og Second (20 min)."
    assert "First" in composed["facts"] and "Second" in composed["facts"]
    assert all(r["message"] in composed["fallback"] for r in fired)


def test_event_inside_several_lead_windows_fires_once(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[1440, 60],
    )
    # Booked half an hour ahead: inside BOTH windows -> one push, not two
    # identical "in 30 min" messages.
    _event_in(settings, "Flight", minutes=30)
    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert fired[0]["lead_minutes"] == 60  # reported at the tightest lead
    # Both leads are claimed together, so no later tick can fire a duplicate.
    assert {r["lead_minutes"] for r in _ledger_rows(settings)} == {60, 1440}
    assert reminders.run_reminders(settings) == []


# --- repeat mode ---------------------------------------------------------- #


def test_repeat_fires_each_band_until_start(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=60)).isoformat(timespec="seconds")
    store.create_event(settings, title="Dentist", start=start)

    messages: list[str] = []
    # Walk wall-clock from 60 min out to the start in 15-min steps.
    for step in range(0, 61, 15):
        monkeypatch.setattr(reminders, "now", lambda s, t=base + timedelta(minutes=step): t)
        messages += [r["message"] for r in reminders.run_reminders(settings)]

    # One nudge per 15-min band: 60, 45, 30, 15, 0 min out.
    assert len(messages) == 5
    assert all("Dentist" in m for m in messages)
    for m, countdown in zip(messages, ["1 hour", "45 min", "30 min", "15 min", "now"], strict=True):
        assert countdown in m
    assert {r["lead_minutes"] for r in _ledger_rows(settings)} == {60, 45, 30, 15, 0}


def test_repeat_same_band_is_idempotent(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=40)).isoformat(timespec="seconds")
    store.create_event(settings, title="Call", start=start)

    monkeypatch.setattr(reminders, "now", lambda s: base)
    assert len(reminders.run_reminders(settings)) == 1  # remaining 40 -> slot 30
    assert reminders.run_reminders(settings) == []  # same band, already claimed


def test_repeat_silent_after_start(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=10)).isoformat(timespec="seconds")
    store.create_event(settings, title="Gone", start=start)

    monkeypatch.setattr(reminders, "now", lambda s: base + timedelta(minutes=25))
    assert reminders.run_reminders(settings) == []  # 15 min past start -> nothing


def test_repeat_at_start_band_fires_once(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    start = (base + timedelta(minutes=10)).isoformat(timespec="seconds")
    store.create_event(settings, title="Kickoff", start=start)

    # The tick lands 40s after the start (ticker jitter): one "starting now".
    monkeypatch.setattr(reminders, "now", lambda s: base + timedelta(minutes=10, seconds=40))
    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert "now" in fired[0]["message"]
    # Next tick, still inside the grace window: the band is claimed, no repeat.
    monkeypatch.setattr(reminders, "now", lambda s: base + timedelta(minutes=11, seconds=40))
    assert reminders.run_reminders(settings) == []


def test_repeat_skip_occurrence_stops_remaining_nudges(tmp_path, monkeypatch) -> None:
    # Regression for the "I'm sick today" incident: after "Exercise in 30 min"
    # fired, skipping today's occurrence (what the agent does when the user
    # declines) must silence the rest of the countdown — the ledger only
    # dedupes, it must not keep the schedule alive past the EXDATE.
    from assistant.calendar import ops

    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        reminder_lead_minutes=[60],
        reminder_repeat_minutes=15,
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    occ = base + timedelta(minutes=30)
    dtstart = (occ - timedelta(days=3)).isoformat(timespec="seconds")
    event = store.create_event(settings, title="Exercise", start=dtstart, rrule="FREQ=DAILY")

    monkeypatch.setattr(reminders, "now", lambda s: base)
    fired = reminders.run_reminders(settings)
    assert len(fired) == 1
    assert "Exercise" in fired[0]["message"] and "30 min" in fired[0]["message"]

    assert ops.apply_op(
        settings, {"op": "skip", "id": event.id, "occurrence": occ.isoformat()}
    ) is not None
    for step in (14, 25, 29):  # the "in 14 min" nudge and every later band
        monkeypatch.setattr(reminders, "now", lambda s, t=base + timedelta(minutes=30 - step): t)
        assert reminders.run_reminders(settings) == []

    # Tomorrow's occurrence is untouched and nudges normally.
    tomorrow = base + timedelta(days=1)
    monkeypatch.setattr(reminders, "now", lambda s: tomorrow)
    refired = reminders.run_reminders(settings)
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
    reminders.run_reminders(settings)  # prunes before firing
    assert any(r["event_id"] == "fresh" for r in _ledger_rows(settings))


# --- proactive loop-in ------------------------------------------------------ #


class _RecordingAgent:
    def __init__(self) -> None:
        self.recorded: list[tuple[str, str]] = []

    def update_state(self, config, update, as_node=None) -> None:
        self.recorded.append(
            (config["configurable"]["thread_id"], update["messages"][0].content)
        )


def test_delivered_reminder_is_recorded_on_threads(settings, monkeypatch) -> None:
    settings.telegram_bot_token = "tok"
    settings.telegram_allowed_chat_ids = [7]
    monkeypatch.setattr(reminders, "deliver_reminder", lambda s, r: True)
    _event_in(settings, "Dentist", minutes=30)
    agent = _RecordingAgent()
    fired = reminders.run_reminders(settings, agent)
    assert len(fired) == 1
    # Recorded verbatim as delivered, ⏰ prefix included.
    assert agent.recorded == [("telegram:7", f"⏰ {fired[0]['message']}")]


def test_no_agent_records_nothing(settings) -> None:
    _event_in(settings, "Dentist", minutes=30)
    assert len(reminders.run_reminders(settings)) == 1  # delivery path unchanged


def test_delivered_reminder_also_records_on_slack_threads(settings, monkeypatch) -> None:
    from assistant import threads

    settings.telegram_bot_token = "tok"
    settings.telegram_allowed_chat_ids = [7]
    settings.slack_bot_token = "xoxb-tok"
    settings.slack_notify_channel = "C9"
    # A Slack conversation in the notify channel has spoken to the assistant.
    threads.touch(settings, "slack:C9:U1")

    monkeypatch.setattr(reminders, "deliver_reminder", lambda s, r: True)
    _event_in(settings, "Dentist", minutes=30)
    agent = _RecordingAgent()
    fired = reminders.run_reminders(settings, agent)
    pushed = f"⏰ {fired[0]['message']}"
    assert ("telegram:7", pushed) in agent.recorded
    assert ("slack:C9:U1", pushed) in agent.recorded
