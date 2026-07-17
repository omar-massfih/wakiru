"""Reflection tests — the push log, the deterministic digest, and the LLM pass."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from assistant import mutes, reflect, threads
from assistant.calendar.context import now
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_heartbeat=True,
    )


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )


def _log_push_at(settings, monkeypatch, when, kind="heartbeat", text="checking in"):
    monkeypatch.setattr("assistant.reflect.now", lambda s: when)
    reflect.log_push(settings, kind, text)
    monkeypatch.undo()


# --- push log ----------------------------------------------------------------- #


def test_log_push_roundtrip_and_excerpt_clipping(settings) -> None:
    reflect.log_push(settings, "heartbeat", "  a\nmessy   push " + "x" * 300)
    rows = reflect.recent_pushes(settings, now(settings) - timedelta(hours=1))
    assert len(rows) == 1
    assert rows[0]["kind"] == "heartbeat"
    assert "\n" not in rows[0]["excerpt"] and len(rows[0]["excerpt"]) <= 120


def test_recent_pushes_filters_by_window(settings, monkeypatch) -> None:
    current = now(settings)
    _log_push_at(settings, monkeypatch, current - timedelta(days=3), text="old push")
    _log_push_at(settings, monkeypatch, current - timedelta(hours=1), text="new push")
    rows = reflect.recent_pushes(settings, current - timedelta(hours=24))
    assert [r["excerpt"] for r in rows] == ["new push"]


def test_log_push_never_raises(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.followups._connect", lambda s: (_ for _ in ()).throw(OSError("boom"))
    )
    reflect.log_push(settings, "heartbeat", "still fine")  # must not raise


# --- the digest --------------------------------------------------------------- #


def test_digest_marks_answered_and_ignored_pushes(settings, monkeypatch) -> None:
    current = now(settings)
    _log_push_at(settings, monkeypatch, current - timedelta(hours=5), text="ignored push")
    _log_push_at(settings, monkeypatch, current - timedelta(minutes=30), text="answered push")
    threads.touch(settings, "telegram:7", user=True, assistant=False)
    digest = reflect.build_digest(settings, current)
    assert '"answered push" — the user wrote back within 2h' in digest
    # The later user contact makes earlier silence ambiguous, never "ignored".
    assert '"ignored push" — the user was active later' in digest


def test_digest_includes_mutes_with_reasons(settings) -> None:
    current = now(settings)
    mutes.set_mute(
        settings, "all", "", current + timedelta(hours=8), "user is sick", current
    )
    digest = reflect.build_digest(settings, current)
    assert "Reminders muted (scope: all)" in digest
    assert 'stated reason: "user is sick"' in digest


def test_digest_includes_undone_calendar_writes(settings) -> None:
    from assistant.calendar.undo import _SPEC
    from assistant.write_ledger import connect, record_write

    current = now(settings)
    record_write(
        _SPEC, settings, "telegram:7", "b1", "ev1", "create",
        "Created 'Dentist' tomorrow 10:00", None,
    )
    with connect(_SPEC, settings) as conn:
        conn.execute(
            "UPDATE write_log SET undone_at = ?",
            (current.isoformat(timespec="seconds"),),
        )
    digest = reflect.build_digest(settings, current)
    assert "The user undid a calendar write: Created 'Dentist' tomorrow 10:00" in digest


def test_empty_digest_for_a_quiet_day(settings) -> None:
    assert reflect.build_digest(settings, now(settings)) == ""


# --- the LLM pass ------------------------------------------------------------- #


def test_reflection_skips_llm_on_empty_digest(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.llm.complete_text",
        lambda *a, **k: pytest.fail("no LLM call on an empty digest"),
    )
    result = reflect.run_reflection(settings)
    assert result == {"ran": False, "reason": "nothing to review"}


def test_reflection_disabled_by_flag(settings) -> None:
    off = Settings(
        memory_dir=settings.memory_dir, timezone="Europe/Oslo", enable_reflection=False
    )
    assert reflect.run_reflection(off) == {"ran": False, "reason": "disabled"}


def test_reflection_saves_self_tagged_procedural_notes(settings, monkeypatch) -> None:
    reflect.log_push(settings, "heartbeat", "evening nudge")
    ops = json.dumps(
        [
            {
                "op": "save",
                "description": "Hold evening ambient pushes",
                "body": "Ambient evening pushes go unanswered; prefer mornings.",
                "kind": "semantic",  # forced to procedural regardless
            }
        ]
    )
    monkeypatch.setattr("assistant.llm.complete_text", lambda *a, **k: ops)
    result = reflect.run_reflection(settings)
    assert result["ran"] and len(result["applied"]) == 1
    notes = reflect.self_notes(settings)
    assert len(notes) == 1
    assert notes[0].kind == "procedural" and "self" in notes[0].tags


def test_reflection_caps_applied_ops(settings, monkeypatch) -> None:
    reflect.log_push(settings, "heartbeat", "a push")
    ops = json.dumps(
        [
            {"op": "save", "description": f"lesson {i}", "body": f"lesson body {i}"}
            for i in range(6)
        ]
    )
    monkeypatch.setattr("assistant.llm.complete_text", lambda *a, **k: ops)
    result = reflect.run_reflection(settings)
    assert len(result["applied"]) == settings.reflection_max_ops


def test_reflection_survives_llm_failure(settings, monkeypatch) -> None:
    reflect.log_push(settings, "heartbeat", "a push")
    monkeypatch.setattr(
        "assistant.llm.complete_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model down")),
    )
    assert reflect.run_reflection(settings) == {"ran": False, "reason": "llm failed"}


# --- surfacing ---------------------------------------------------------------- #


def test_heartbeat_prefix_carries_self_notes(settings, monkeypatch) -> None:
    from langchain_core.messages import AIMessage

    from assistant import heartbeat
    from assistant.memory.learn import save_memory

    save_memory(
        settings,
        body="Prefer mornings for ambient pushes.",
        kind="procedural",
        source="reflection",
        tags=["self"],
    )

    class _Model:
        def __init__(self):
            self.prompts = []

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.prompts.append(list(messages))
            return AIMessage(content="SILENT")

    model = _Model()
    monkeypatch.setattr(heartbeat, "build_model", lambda s=None: model)
    heartbeat.run_heartbeat(settings)
    joined = "\n".join(str(m.content) for m in model.prompts[0])
    assert "What you have learned about your own proactivity" in joined
    assert "Prefer mornings for ambient pushes." in joined


def test_sleep_pass_reports_reflection(settings, monkeypatch) -> None:
    from assistant import sleep

    monkeypatch.setattr(
        "assistant.reflect.run_reflection",
        lambda s, c=None: {"ran": False, "reason": "nothing to review"},
    )
    result = sleep.run_sleep(settings, force=True)
    assert result["reflection"] == {"ran": False, "reason": "nothing to review"}
