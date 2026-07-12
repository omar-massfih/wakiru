"""Daily briefing tests — the due-time gate, the once-per-day ledger, delivery.

No network and no LLM: delivery and the polish call are monkeypatched. The
ledger runs for real against a tmp SQLite file.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from assistant import briefing
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_briefing=True,
        briefing_llm_polish=False,  # keep run_codex out of these tests
        enable_email=False,
    )


@pytest.fixture
def delivered(monkeypatch) -> list[dict]:
    sent: list[dict] = []
    monkeypatch.setattr(
        briefing, "deliver_reminder", lambda s, reminder: sent.append(reminder) or True
    )
    return sent


def _freeze_clock(monkeypatch, settings: Settings, hhmm: str) -> None:
    hour, minute = map(int, hhmm.split(":"))
    from assistant.calendar.context import resolve_tz

    frozen = datetime(2026, 7, 11, hour, minute, tzinfo=resolve_tz(settings))
    monkeypatch.setattr(briefing, "now", lambda s: frozen)


def test_not_due_before_briefing_time(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, "06:00")
    assert briefing.run_briefing(settings) == {"sent": False, "reason": "not due yet"}
    assert delivered == []


def test_fires_once_after_due_time(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, "08:00")
    first = briefing.run_briefing(settings)
    assert first["sent"] and first["delivered"]
    assert len(delivered) == 1
    assert delivered[0]["title"] == "Daily briefing"
    assert "Upcoming events" in delivered[0]["message"]

    second = briefing.run_briefing(settings)
    assert second == {"sent": False, "reason": "already sent today"}
    assert len(delivered) == 1


def test_disabled_is_noop_unless_forced(settings, delivered, monkeypatch) -> None:
    settings.enable_briefing = False
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings) == {"sent": False, "reason": "disabled"}
    assert briefing.run_briefing(settings, force=True)["sent"]
    assert len(delivered) == 1


def test_force_skips_time_gate_but_claims_the_day(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, "06:00")
    assert briefing.run_briefing(settings, force=True)["sent"]
    # The scheduled firing later the same day must not duplicate it.
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings)["reason"] == "already sent today"
    assert len(delivered) == 1


def test_polish_failure_falls_back_to_raw_digest(settings, delivered, monkeypatch) -> None:
    settings.briefing_llm_polish = True

    def boom(prompt, settings=None):
        raise RuntimeError("codex unavailable")

    monkeypatch.setattr("assistant.codex_runner.run_codex", boom)
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings)["sent"]
    assert "Upcoming events" in delivered[0]["message"]


def test_malformed_briefing_time_defaults(settings) -> None:
    settings.briefing_time = "not-a-time"
    assert briefing._due_time(settings).hour == 7
