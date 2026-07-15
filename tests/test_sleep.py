"""Clock-driven sleep tests — the time gate, the once-per-day ledger, the LLM
idle gate, the fold sweep, and that quiet hours do not hold it.

Consolidation is faked (a spy recording ``include_llm``) so no real LLM runs;
the ledger, the state KV, the note store, and the thread registry all run for
real against tmp_path SQLite.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from assistant import sleep, threads
from assistant.calendar.context import resolve_tz
from assistant.config import Settings
from assistant.memory import store
from assistant.memory.store import Note


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_sleep=True,
        reminder_webhook_url=None,
    )


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )


@pytest.fixture
def spy_consolidate(monkeypatch) -> list[bool]:
    """Replace consolidate_memory with a spy recording each include_llm flag."""
    calls: list[bool] = []

    def _fake(settings, *, include_llm=True):
        calls.append(include_llm)
        return {"pruned_episodes": 0, "changes": []}

    monkeypatch.setattr(sleep, "consolidate_memory", _fake)
    return calls


def _at(settings: Settings, hhmm: str) -> datetime:
    hour, minute = map(int, hhmm.split(":"))
    return datetime(2026, 7, 11, hour, minute, tzinfo=resolve_tz(settings))


def _clock(monkeypatch, settings: Settings, hhmm: str) -> None:
    monkeypatch.setattr(sleep, "now", lambda s: _at(settings, hhmm))


def _episode(settings: Settings, name: str, updated: str) -> None:
    store.write_note(
        settings,
        Note(name=name, description=name, body="x", kind="episodic", updated=updated),
    )


# --- the time-of-day and once-per-day gates ---------------------------------- #


def test_not_due_before_sleep_time(settings, monkeypatch, spy_consolidate) -> None:
    _clock(monkeypatch, settings, "02:00")  # before the 03:30 default
    result = sleep.run_sleep(settings)
    assert result == {"ran": False, "reason": "not due yet"}
    assert spy_consolidate == []


def test_claims_once_per_day(settings, monkeypatch, spy_consolidate) -> None:
    _clock(monkeypatch, settings, "04:00")
    first = sleep.run_sleep(settings)
    assert first["ran"] is True
    second = sleep.run_sleep(settings)
    assert second == {"ran": False, "reason": "already ran today"}
    assert spy_consolidate == [True]  # only the first pass consolidated


def test_disabled_is_a_noop(settings, monkeypatch, spy_consolidate) -> None:
    off = settings.model_copy(update={"enable_sleep": False})
    _clock(monkeypatch, off, "04:00")
    assert sleep.run_sleep(off) == {"ran": False, "reason": "disabled"}
    assert spy_consolidate == []


def test_force_bypasses_time_gate_but_still_claims(
    settings, monkeypatch, spy_consolidate
) -> None:
    _clock(monkeypatch, settings, "02:00")  # well before due
    forced = sleep.run_sleep(settings, force=True)
    assert forced["ran"] is True
    # Same day: the forced claim blocks a later scheduled run.
    _clock(monkeypatch, settings, "04:00")
    assert sleep.run_sleep(settings) == {"ran": False, "reason": "already ran today"}
    assert spy_consolidate == [True]


def test_runs_during_quiet_hours(settings, monkeypatch, spy_consolidate) -> None:
    # 03:30 sits inside the default quiet window; sleep must run anyway (it never
    # pushes). Make in_quiet_hours explode if consulted to prove it is not a gate.
    monkeypatch.setattr(
        "assistant.memory.profile.in_quiet_hours",
        lambda *a, **k: pytest.fail("sleep must not consult quiet hours"),
    )
    _clock(monkeypatch, settings, "03:30")
    assert sleep.run_sleep(settings)["ran"] is True
    assert spy_consolidate == [True]


# --- the LLM idle gate ------------------------------------------------------- #


def test_llm_runs_on_first_pass(settings, monkeypatch, spy_consolidate) -> None:
    _episode(settings, "ep-1", "2026-07-11")
    _clock(monkeypatch, settings, "04:00")
    result = sleep.run_sleep(settings)
    assert result["llm"] is True and spy_consolidate == [True]


def test_llm_skipped_when_no_new_episode(settings, monkeypatch, spy_consolidate) -> None:
    # First night, with an episode dated the same day → LLM runs and records
    # the pass date. Second night with nothing new → LLM skipped.
    _episode(settings, "ep-1", "2026-07-11")
    _clock(monkeypatch, settings, "04:00")  # local date 2026-07-11
    assert sleep.run_sleep(settings)["llm"] is True

    monkeypatch.setattr(sleep, "now", lambda s: _at(settings, "04:00").replace(day=12))
    second = sleep.run_sleep(settings)
    assert second["ran"] is True and second["llm"] is False
    assert spy_consolidate == [True, False]


def test_llm_runs_again_when_a_newer_episode_arrives(
    settings, monkeypatch, spy_consolidate
) -> None:
    _episode(settings, "ep-1", "2026-07-11")
    _clock(monkeypatch, settings, "04:00")
    assert sleep.run_sleep(settings)["llm"] is True

    _episode(settings, "ep-2", "2026-07-12")  # a fresh episode next day
    monkeypatch.setattr(sleep, "now", lambda s: _at(settings, "04:00").replace(day=12))
    assert sleep.run_sleep(settings)["llm"] is True
    assert spy_consolidate == [True, True]


# --- the fold sweep ---------------------------------------------------------- #


def test_fold_sweep_touches_every_known_thread(
    settings, monkeypatch, spy_consolidate
) -> None:
    threads.touch(settings, "telegram:1")
    threads.touch(settings, "slack:2")
    folded: list[str] = []
    monkeypatch.setattr(sleep, "maybe_summarize", lambda a, s, tid: folded.append(tid))
    _clock(monkeypatch, settings, "04:00")
    result = sleep.run_sleep(settings, agent=object())
    assert result["folded"] == 2
    assert set(folded) == {"telegram:1", "slack:2"}


def test_fold_sweep_is_skipped_without_an_agent(
    settings, monkeypatch, spy_consolidate
) -> None:
    threads.touch(settings, "telegram:1")
    _clock(monkeypatch, settings, "04:00")
    result = sleep.run_sleep(settings)  # no agent
    assert result["folded"] == 0
