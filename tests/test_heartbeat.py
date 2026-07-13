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
