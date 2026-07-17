"""Importance-classification tests — LLM grading, the verdict cache, fallback.

Everything runs for real (plain SQLite); faked is only the LLM call
(``assistant.llm.complete_text``), so these stay fast and offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from assistant.calendar import context, importance, store
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), timezone="Europe/Oslo")


def _event(settings: Settings, title: str) -> store.Event:
    start = (context.now(settings) + timedelta(days=1)).isoformat(timespec="seconds")
    return store.create_event(settings, title=title, start=start)


def _stub_llm(monkeypatch, reply) -> list[str]:
    """Patch complete_text; returns the list of prompts it received."""
    prompts: list[str] = []

    def fake(prompt, settings=None, *, system=None):
        prompts.append(prompt)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("assistant.llm.complete_text", fake)
    return prompts


def _rows(settings: Settings) -> dict[str, dict]:
    with importance._connect(settings) as conn:
        return {
            row["event_id"]: dict(row)
            for row in conn.execute("SELECT * FROM event_importance")
        }


# --- leads / horizon ------------------------------------------------------- #


def test_leads_for_maps_tiers(settings) -> None:
    assert importance.leads_for(settings, importance.TIER_CRITICAL) == [2880, 1440, 180, 60, 15]
    assert importance.leads_for(settings, importance.TIER_NORMAL) == [15]
    assert importance.max_lead_minutes(settings) == 2880


# --- classification + cache ------------------------------------------------ #


def test_llm_verdicts_are_applied_and_cached(settings, monkeypatch) -> None:
    doctor = _event(settings, "Legetime hos Dr. Berg")
    coffee = _event(settings, "Coffee with Anna")
    prompts = _stub_llm(
        monkeypatch, f'{{"{doctor.id}": "critical", "{coffee.id}": "normal"}}'
    )

    tiers = importance.tiers_for(settings, [doctor, coffee])
    assert tiers == {doctor.id: "critical", coffee.id: "normal"}
    assert len(prompts) == 1  # both events graded in ONE batched call
    assert doctor.title in prompts[0] and coffee.title in prompts[0]
    assert {r["source"] for r in _rows(settings).values()} == {"llm"}

    # Cached: a second pass must not call the model at all.
    monkeypatch.setattr(
        "assistant.llm.complete_text",
        lambda *a, **k: pytest.fail("cached verdicts must not re-call the LLM"),
    )
    assert importance.tiers_for(settings, [doctor, coffee]) == tiers


def test_cache_survives_restart(settings, monkeypatch) -> None:
    doctor = _event(settings, "Tannlege")
    _stub_llm(monkeypatch, f'{{"{doctor.id}": "critical"}}')
    importance.tiers_for(settings, [doctor])

    # A fresh Settings over the same data dir sees the persisted verdict.
    reopened = Settings(memory_dir=settings.memory_dir, timezone="Europe/Oslo")
    monkeypatch.setattr(
        "assistant.llm.complete_text",
        lambda *a, **k: pytest.fail("persisted verdicts must not re-call the LLM"),
    )
    assert importance.tiers_for(reopened, [doctor]) == {doctor.id: "critical"}


def test_title_change_reclassifies(settings, monkeypatch) -> None:
    event = _event(settings, "Katta til veterinær")
    _stub_llm(monkeypatch, f'{{"{event.id}": "normal"}}')
    assert importance.tiers_for(settings, [event]) == {event.id: "normal"}

    renamed = store.update_event(settings, event.id, title="Operasjon på sykehuset")
    prompts = _stub_llm(monkeypatch, f'{{"{event.id}": "critical"}}')
    assert importance.tiers_for(settings, [renamed]) == {event.id: "critical"}
    assert len(prompts) == 1  # hash miss => regraded


def test_json_is_extracted_from_chatty_reply(settings, monkeypatch) -> None:
    event = _event(settings, "Fly til Bergen")
    _stub_llm(
        monkeypatch,
        f'Sure! Here is the grading:\n{{"{event.id}": "critical"}}\nLet me know!',
    )
    assert importance.tiers_for(settings, [event]) == {event.id: "critical"}


# --- fallback -------------------------------------------------------------- #


def test_llm_failure_falls_back_to_normal(settings, monkeypatch) -> None:
    event = _event(settings, "Legetime")
    _stub_llm(monkeypatch, RuntimeError("model down"))
    assert importance.tiers_for(settings, [event]) == {event.id: "normal"}
    assert _rows(settings)[event.id]["source"] == "fallback"


def test_malformed_reply_falls_back_to_normal(settings, monkeypatch) -> None:
    event = _event(settings, "Legetime")
    _stub_llm(monkeypatch, "I cannot help with that.")
    assert importance.tiers_for(settings, [event]) == {event.id: "normal"}
    assert _rows(settings)[event.id]["source"] == "fallback"


def test_fallback_not_retried_inside_backoff(settings, monkeypatch) -> None:
    event = _event(settings, "Legetime")
    _stub_llm(monkeypatch, RuntimeError("model down"))
    importance.tiers_for(settings, [event])

    # Still inside the backoff window: no second call even though the model is up.
    monkeypatch.setattr(
        "assistant.llm.complete_text",
        lambda *a, **k: pytest.fail("fallback must not retry inside the backoff"),
    )
    assert importance.tiers_for(settings, [event]) == {event.id: "normal"}


def test_fallback_retried_after_backoff(settings, monkeypatch) -> None:
    event = _event(settings, "Legetime")
    _stub_llm(monkeypatch, RuntimeError("model down"))
    importance.tiers_for(settings, [event])

    # Age the fallback row past the backoff; the recovered model regrades it.
    aged = (
        datetime.now().astimezone() - importance.FALLBACK_RETRY - timedelta(minutes=1)
    ).isoformat(timespec="seconds")
    with importance._connect(settings) as conn:
        conn.execute("UPDATE event_importance SET updated = ?", (aged,))
    _stub_llm(monkeypatch, f'{{"{event.id}": "critical"}}')
    assert importance.tiers_for(settings, [event]) == {event.id: "critical"}
    assert _rows(settings)[event.id]["source"] == "llm"


def test_id_missing_from_reply_stays_retryable(settings, monkeypatch) -> None:
    graded = _event(settings, "Eksamen")
    skipped = _event(settings, "Middag med Nora")
    _stub_llm(monkeypatch, f'{{"{graded.id}": "critical"}}')
    tiers = importance.tiers_for(settings, [graded, skipped])
    assert tiers == {graded.id: "critical", skipped.id: "normal"}
    rows = _rows(settings)
    assert rows[graded.id]["source"] == "llm"
    assert rows[skipped.id]["source"] == "fallback"  # retried next window


# --- pruning --------------------------------------------------------------- #


def test_old_verdicts_are_pruned_on_write(settings, monkeypatch) -> None:
    old = (
        datetime.now().astimezone()
        - timedelta(days=importance.RETENTION_DAYS + 1)
    ).isoformat(timespec="seconds")
    with importance._connect(settings) as conn:
        conn.execute(
            "INSERT INTO event_importance (event_id, title_hash, tier, source, updated)"
            " VALUES ('stale', 'x', 'normal', 'llm', ?)",
            (old,),
        )
    event = _event(settings, "Legetime")
    _stub_llm(monkeypatch, f'{{"{event.id}": "critical"}}')
    importance.tiers_for(settings, [event])
    assert "stale" not in _rows(settings)
