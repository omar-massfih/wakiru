"""Heartbeat tests — the holds, the every-beat wake, judgment, and delivery.

The model is faked (a scripted tool-calling chat model, as in test_agent.py);
everything else — the followup store, mutes, quiet hours, the state KV —
runs for real against tmp_path SQLite.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage

from assistant import followups, heartbeat, threads
from assistant.calendar.context import now
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_heartbeat=True,
        reminder_webhook_url=None,
    )


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )


class _ScriptedModel:
    """A fake chat model replying with a fixed script of AIMessages."""

    def __init__(self, script: list[AIMessage]) -> None:
        self.script = list(script)
        self.prompts: list[list] = []
        self.bound_schemas: list[dict] = []

    def bind_tools(self, tools):
        self.bound_schemas = list(tools)
        return self

    def invoke(self, messages):
        self.prompts.append(list(messages))
        return self.script.pop(0)


def _wire_model(monkeypatch, script: list[AIMessage]) -> _ScriptedModel:
    model = _ScriptedModel(script)
    monkeypatch.setattr(heartbeat, "build_model", lambda s=None: model)
    return model


def _due_followup(settings: Settings, topic: str = "ask about the interview") -> None:
    followups.add(settings, now(settings) - timedelta(minutes=5), topic, "It was at NAV")


# --- gather_situation: holds and the situation report ------------------------- #


def test_trigger_less_beat_still_wakes_the_model(settings, monkeypatch) -> None:
    # Nothing happened — the model is still woken to judge, and SILENT
    # (the normal outcome) delivers nothing.
    model = _wire_model(monkeypatch, [AIMessage(content="SILENT")])
    monkeypatch.setattr(
        "assistant.notify.deliver_reminder",
        lambda *a, **k: pytest.fail("SILENT must not deliver"),
    )
    result = heartbeat.run_heartbeat(settings)
    assert result["sent"] is False and result["reason"] == "silent"
    joined = "\n".join(str(m.content) for m in model.prompts[0])
    assert "Nothing specific happened" in joined


def test_disabled_never_wakes(settings) -> None:
    off = settings.model_copy(update={"enable_heartbeat": False})
    _due_followup(off)
    assert heartbeat.gather_situation(off) is None
    assert followups.list_open(off)  # and nothing was claimed


def test_quiet_hours_hold_without_claiming(settings, monkeypatch) -> None:
    _due_followup(settings)
    monkeypatch.setattr("assistant.memory.profile.in_quiet_hours", lambda s, c: True)
    assert heartbeat.gather_situation(settings) is None
    assert followups.list_open(settings)  # still open — raised after quiet ends


def test_all_mute_holds_without_claiming(settings) -> None:
    from assistant.mutes import set_mute

    _due_followup(settings)
    set_mute(settings, "all", "", now(settings) + timedelta(hours=2))
    assert heartbeat.gather_situation(settings) is None
    assert followups.list_open(settings)


def test_due_followup_is_claimed_exactly_once(settings) -> None:
    _due_followup(settings)
    situation = heartbeat.gather_situation(settings)
    assert situation is not None
    assert [f.topic for f in situation.followups] == ["ask about the interview"]
    assert "It was at NAV" in situation.report()
    assert situation.scheduled
    again = heartbeat.gather_situation(settings)  # consumed — next beat is ambient
    assert again is not None and again.followups == [] and not again.scheduled


def test_contact_staleness_is_reported(settings) -> None:
    stale = settings.model_copy(update={"heartbeat_contact_gap_hours": 24})
    threads.touch(stale, "telegram:7")
    # Fresh contact: no staleness line, but the beat still gathers.
    fresh = heartbeat.gather_situation(stale)
    assert fresh is not None
    assert "haven't heard from the user" not in fresh.report()
    assert "last heard from the user" in fresh.report()  # ambient fact, always
    # Backdate the contact stamp two days.
    old = (now(stale) - timedelta(days=2)).isoformat(timespec="seconds")
    with threads._connect(stale) as conn:
        conn.execute("UPDATE known_threads SET last_user_at = ?", (old,))
    situation = heartbeat.gather_situation(stale)
    assert situation is not None and "haven't heard from the user" in situation.report()


def test_mail_change_is_reported_once_per_snapshot(settings, monkeypatch) -> None:
    mail_on = settings.model_copy(update={"enable_email": True})
    monkeypatch.setattr(
        "assistant.mail.snapshot.current",
        lambda s: "## Unread mail (snapshot as of 09:12)\n1 unread message(s):\n- Hei",
    )
    first = heartbeat.gather_situation(mail_on)
    assert first is not None and "unread-mail snapshot changed" in first.report()
    # The same snapshot never re-raises (hash consumed) — later beats gather
    # without the mail line.
    second = heartbeat.gather_situation(mail_on)
    assert second is not None
    assert "unread-mail snapshot changed" not in second.report()


def test_open_followups_are_surfaced_before_they_are_due(settings) -> None:
    # A future (not-yet-due) followup is the assistant's standing intention —
    # it must appear in every situation report so the model can act on it,
    # revise it, or let it ride, without waiting for it to fire.
    followups.add(
        settings,
        now(settings) + timedelta(hours=6),
        "chase the apartment reply",
        "waiting on the landlord",
    )
    situation = heartbeat.gather_situation(settings)
    assert situation is not None
    report = situation.report()
    assert "Open follow-ups you are carrying" in report
    assert "chase the apartment reply" in report
    assert "waiting on the landlord" in report  # the note-to-self rides along
    assert not situation.followups  # not due, so not claimed — still open
    assert followups.list_open(settings)


def test_revised_followup_context_shows_on_the_next_beat(settings, monkeypatch) -> None:
    # A beat that updates a followup's context; the next beat's report reflects it.
    f = followups.add(
        settings, now(settings) + timedelta(hours=6), "chase reply", "no word yet"
    )
    _wire_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "update_followup",
                    "args": {"target": f.id, "context": "landlord replied, sending docs"},
                    "id": "u1",
                }],
            ),
            AIMessage(content="SILENT"),
        ],
    )
    heartbeat.run_heartbeat(settings)
    assert followups.list_open(settings)[0].context == "landlord replied, sending docs"
    later = heartbeat.gather_situation(settings)
    assert later is not None and "landlord replied, sending docs" in later.report()


# --- run_heartbeat: the wake loop and delivery -------------------------------- #


def test_silent_verdict_delivers_nothing(settings, monkeypatch) -> None:
    _due_followup(settings)
    _wire_model(monkeypatch, [AIMessage(content="SILENT")])
    monkeypatch.setattr(
        "assistant.notify.deliver_reminder",
        lambda *a, **k: pytest.fail("SILENT must not deliver"),
    )
    result = heartbeat.run_heartbeat(settings)
    assert result["sent"] is False and result["reason"] == "silent"
    assert "followup: ask about the interview" in result["triggers"]


def test_message_is_delivered_and_looped_in(settings, monkeypatch) -> None:
    _due_followup(settings)
    _wire_model(monkeypatch, [AIMessage(content="Hei! Hvordan gikk intervjuet?")])
    delivered: list[dict] = []
    recorded: list[str] = []
    monkeypatch.setattr(
        "assistant.notify.deliver_reminder", lambda s, r: delivered.append(r) or True
    )
    monkeypatch.setattr(
        "assistant.proactive.record_push", lambda agent, s, text: recorded.append(text)
    )
    result = heartbeat.run_heartbeat(settings, agent=object())
    assert result["sent"] is True and result["delivered"] is True
    assert delivered == [{"title": "Wakiru", "message": "Hei! Hvordan gikk intervjuet?"}]
    assert recorded == ["Hei! Hvordan gikk intervjuet?"]


def test_ambient_push_is_throttled_by_min_gap(settings, monkeypatch) -> None:
    # A purely ambient wake (no followup, no briefing) whose push would land
    # within the min gap of the previous push is suppressed — the bound is on
    # delivery, never on the model's judgment.
    _wire_model(monkeypatch, [AIMessage(content="Tenkte på deg!")])
    monkeypatch.setattr(
        "assistant.notify.deliver_reminder",
        lambda *a, **k: pytest.fail("a throttled push must not deliver"),
    )
    heartbeat._state_set(
        settings, "last_push_at", now(settings).isoformat(timespec="seconds")
    )
    result = heartbeat.run_heartbeat(settings)
    assert result["sent"] is False and result["reason"] == "throttled"


def test_ambient_push_delivers_once_the_gap_has_passed(settings, monkeypatch) -> None:
    _wire_model(monkeypatch, [AIMessage(content="Tenkte på deg!")])
    delivered: list[dict] = []
    monkeypatch.setattr(
        "assistant.notify.deliver_reminder", lambda s, r: delivered.append(r) or True
    )
    old = now(settings) - timedelta(minutes=settings.heartbeat_min_gap_minutes + 1)
    heartbeat._state_set(settings, "last_push_at", old.isoformat(timespec="seconds"))
    result = heartbeat.run_heartbeat(settings)
    assert result["sent"] is True and delivered


def test_scheduled_intent_delivers_regardless_of_gap(settings, monkeypatch) -> None:
    _due_followup(settings)
    _wire_model(monkeypatch, [AIMessage(content="Hvordan gikk intervjuet?")])
    delivered: list[dict] = []
    monkeypatch.setattr(
        "assistant.notify.deliver_reminder", lambda s, r: delivered.append(r) or True
    )
    heartbeat._state_set(
        settings, "last_push_at", now(settings).isoformat(timespec="seconds")
    )
    result = heartbeat.run_heartbeat(settings)
    assert result["sent"] is True and delivered


def test_force_bypasses_the_ambient_throttle(settings, monkeypatch) -> None:
    _wire_model(monkeypatch, [AIMessage(content="Tenkte på deg!")])
    delivered: list[dict] = []
    monkeypatch.setattr(
        "assistant.notify.deliver_reminder", lambda s, r: delivered.append(r) or True
    )
    heartbeat._state_set(
        settings, "last_push_at", now(settings).isoformat(timespec="seconds")
    )
    result = heartbeat.run_heartbeat(settings, force=True)
    assert result["sent"] is True and delivered


def test_wake_prompt_carries_persona_context_and_situation(settings, monkeypatch) -> None:
    _due_followup(settings)
    model = _wire_model(monkeypatch, [AIMessage(content="SILENT")])
    heartbeat.run_heartbeat(settings)

    prompt = model.prompts[0]
    joined = "\n".join(str(m.content) for m in prompt)
    assert "You are Wakiru" in joined  # persona leads
    assert "Situation report" in joined and "ask about the interview" in joined
    assert "scheduled background wake" in joined  # the instruction
    assert "Current date and time" in joined  # context providers ran


def test_wake_can_use_tools_before_answering(settings, monkeypatch) -> None:
    _due_followup(settings)
    when = (now(settings) + timedelta(days=1)).isoformat(timespec="seconds")
    model = _wire_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "schedule_followup",
                    "args": {"when": when, "topic": "check again tomorrow"},
                    "id": "c1",
                }],
            ),
            AIMessage(content="SILENT"),
        ],
    )
    result = heartbeat.run_heartbeat(settings)
    assert result["reason"] == "silent"
    assert [f.topic for f in followups.list_open(settings)] == ["check again tomorrow"]
    assert {s["function"]["name"] for s in model.bound_schemas} >= {"schedule_followup"}
    bound = {s["function"]["name"] for s in model.bound_schemas}
    assert "send_email" not in bound and "undo" not in bound


def test_composition_failure_is_contained(settings, monkeypatch) -> None:
    _due_followup(settings)

    class _Boom:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("model down")

    monkeypatch.setattr(heartbeat, "build_model", lambda s=None: _Boom())
    result = heartbeat.run_heartbeat(settings)
    assert result["sent"] is False and result["reason"] == "composition failed"


# --- the briefing as a heartbeat trigger -------------------------------------- #


def _at(settings: Settings, hhmm: str):
    from datetime import datetime

    from assistant.calendar.context import resolve_tz

    hour, minute = map(int, hhmm.split(":"))
    return datetime(2026, 7, 11, hour, minute, tzinfo=resolve_tz(settings))


def test_briefing_trigger_claims_once_per_day(settings) -> None:
    with_briefing = settings.model_copy(update={"enable_briefing": True})
    early = heartbeat.gather_situation(with_briefing, _at(with_briefing, "06:00"))
    assert early is not None  # the beat gathers…
    assert "daily briefing" not in early.report().lower()  # …but no trigger yet

    due = heartbeat.gather_situation(with_briefing, _at(with_briefing, "08:00"))
    assert due is not None and "daily briefing" in due.report().lower()
    assert due.scheduled

    again = heartbeat.gather_situation(with_briefing, _at(with_briefing, "09:00"))
    assert again is not None
    assert "daily briefing" not in again.report().lower()  # claimed for the day


def test_briefing_ledger_is_shared_with_template_path(settings, monkeypatch) -> None:
    # A template-path briefing earlier the same day must block the heartbeat
    # trigger (and vice versa) — same ledger, never a double brief.
    from assistant import briefing

    with_briefing = settings.model_copy(
        update={"enable_briefing": True, "enable_heartbeat": False}
    )
    monkeypatch.setattr(briefing, "deliver_reminder", lambda s, r: True)
    monkeypatch.setattr(briefing, "now", lambda s: _at(with_briefing, "08:00"))
    monkeypatch.setattr(
        "assistant.compose.compose_push", lambda s, **kw: kw["fallback"]
    )
    assert briefing.run_briefing(with_briefing)["sent"]

    both_on = with_briefing.model_copy(update={"enable_heartbeat": True})
    later = heartbeat.gather_situation(both_on, _at(both_on, "09:00"))
    assert later is not None and "daily briefing" not in later.report().lower()


def test_forced_briefing_bypasses_time_gate_only(settings) -> None:
    with_briefing = settings.model_copy(update={"enable_briefing": True})
    early = heartbeat.gather_situation(
        with_briefing, _at(with_briefing, "06:00"), force_briefing=True
    )
    assert early is not None and "daily briefing" in early.report().lower()
    # Still once per day: the forced claim blocks the scheduled one.
    scheduled = heartbeat.gather_situation(with_briefing, _at(with_briefing, "08:00"))
    assert scheduled is not None
    assert "daily briefing" not in scheduled.report().lower()
