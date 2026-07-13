"""Heartbeat tests — LLM-free triage, the holds, the wake loop, and delivery.

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


# --- gather_situation: the LLM-free triage ----------------------------------- #


def test_no_triggers_means_no_wake_and_no_model_call(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        heartbeat, "build_model",
        lambda s=None: pytest.fail("no trigger must mean no model call"),
    )
    result = heartbeat.run_heartbeat(settings)
    assert result == {"sent": False, "reason": "nothing to do"}


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
    assert heartbeat.gather_situation(settings) is None  # consumed


def test_contact_staleness_triggers_and_respects_gap(settings) -> None:
    stale = settings.model_copy(update={"heartbeat_contact_gap_hours": 24})
    threads.touch(stale, "telegram:7")
    # Fresh contact: no trigger.
    assert heartbeat.gather_situation(stale) is None
    # Backdate the contact stamp two days.
    old = (now(stale) - timedelta(days=2)).isoformat(timespec="seconds")
    with threads._connect(stale) as conn:
        conn.execute("UPDATE known_threads SET last_user_at = ?", (old,))
    situation = heartbeat.gather_situation(stale)
    assert situation is not None and "haven't heard from the user" in situation.report()

    # A recent wake throttles the ambient trigger (min-gap) …
    heartbeat._state_set(stale, "last_wake_at", now(stale).isoformat(timespec="seconds"))
    assert heartbeat.gather_situation(stale) is None
    # … but force (the manual endpoint) bypasses the throttle.
    assert heartbeat.gather_situation(stale, force=True) is not None


def test_mail_change_triggers_once_per_snapshot(settings, monkeypatch) -> None:
    mail_on = settings.model_copy(update={"enable_email": True})
    monkeypatch.setattr(
        "assistant.mail.snapshot.current",
        lambda s: "## Unread mail (snapshot as of 09:12)\n1 unread message(s):\n- Hei",
    )
    first = heartbeat.gather_situation(mail_on)
    assert first is not None and "unread-mail snapshot changed" in first.report()
    # The same snapshot never re-triggers (hash consumed), even after the gap.
    heartbeat._state_set(mail_on, "last_wake_at", "")
    assert heartbeat.gather_situation(mail_on) is None


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
    assert "send_email" not in {s["function"]["name"] for s in model.bound_schemas}


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
    assert early is None  # before briefing_time — no trigger

    due = heartbeat.gather_situation(with_briefing, _at(with_briefing, "08:00"))
    assert due is not None and "daily briefing" in due.report().lower()

    again = heartbeat.gather_situation(with_briefing, _at(with_briefing, "09:00"))
    assert again is None  # claimed for the day


def test_briefing_ledger_is_shared_with_template_path(settings, monkeypatch) -> None:
    # A template-path briefing earlier the same day must block the heartbeat
    # trigger (and vice versa) — same ledger, never a double brief.
    from assistant import briefing

    with_briefing = settings.model_copy(
        update={"enable_briefing": True, "enable_heartbeat": False}
    )
    monkeypatch.setattr(briefing, "deliver_reminder", lambda s, r: True)
    monkeypatch.setattr(briefing, "now", lambda s: _at(with_briefing, "08:00"))
    assert briefing.run_briefing(with_briefing)["sent"]

    both_on = with_briefing.model_copy(update={"enable_heartbeat": True})
    assert heartbeat.gather_situation(both_on, _at(both_on, "09:00")) is None


def test_forced_briefing_bypasses_time_gate_only(settings) -> None:
    with_briefing = settings.model_copy(update={"enable_briefing": True})
    early = heartbeat.gather_situation(
        with_briefing, _at(with_briefing, "06:00"), force_briefing=True
    )
    assert early is not None and "daily briefing" in early.report().lower()
    # Still once per day: the forced claim blocks the scheduled one.
    assert heartbeat.gather_situation(with_briefing, _at(with_briefing, "08:00")) is None
