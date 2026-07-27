"""Refresh tests — the data jobs that ride each heartbeat wake.

Each job (mail snapshot, weather, ICS feeds, CalDAV) runs only when its interval
has elapsed since the last stamp, and re-stamps after every attempt so an outage
retries next interval rather than every wake. next_due_at feeds the wake
scheduler so a far-off self-paced wake can't starve them.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant import heartbeat, refreshes
from assistant.calendar.context import now
from assistant.calendar.store import parse_dt
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Weather on with a location so exactly one refresh job is enabled; the
    # worker itself is faked so no network is touched.
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_weather=True,
        weather_latitude=59.9,
        weather_longitude=10.7,
        weather_refresh_minutes=60,
    )


def test_run_due_runs_and_stamps_when_never_run(settings, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("assistant.weather.maybe_refresh", lambda s: calls.append("w"))

    ran = refreshes.run_due(settings)
    assert ran.get("weather") is True
    assert calls == ["w"]
    # The stamp is written, so the next immediate wake is a no-op.
    assert parse_dt(heartbeat.state_get(settings, "refresh:weather")) is not None


def test_run_due_skips_within_the_interval(settings, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("assistant.weather.maybe_refresh", lambda s: calls.append("w"))

    refreshes.run_due(settings)  # first run stamps
    refreshes.run_due(settings)  # within 60 min: skipped
    assert calls == ["w"]


def test_failed_job_restamps_so_it_retries_next_interval(settings, monkeypatch) -> None:
    def boom(s):
        raise RuntimeError("weather API down")

    monkeypatch.setattr("assistant.weather.maybe_refresh", boom)
    ran = refreshes.run_due(settings)
    assert ran.get("weather") is False
    # Stamped despite failing — it retries on the next interval, not next wake.
    assert parse_dt(heartbeat.state_get(settings, "refresh:weather")) is not None


def test_next_due_at_is_now_when_never_run(settings) -> None:
    current = now(settings)
    due = refreshes.next_due_at(settings, current)
    assert due == [current]  # only weather is enabled, and it has never run


def test_next_due_at_is_stamp_plus_interval_after_a_run(settings, monkeypatch) -> None:
    monkeypatch.setattr("assistant.weather.maybe_refresh", lambda s: None)
    refreshes.run_due(settings)
    current = now(settings)
    due = refreshes.next_due_at(settings, current)
    assert len(due) == 1
    # Roughly stamp + 60 min (the stamp was written ~now).
    assert abs(due[0] - (current + timedelta(minutes=60))) < timedelta(seconds=5)


def test_disabled_jobs_are_neither_run_nor_scheduled(tmp_path, monkeypatch) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"))  # weather/email/feeds off
    monkeypatch.setattr(
        "assistant.weather.maybe_refresh", lambda s: pytest.fail("weather is disabled")
    )
    assert refreshes.run_due(settings) == {}
    assert refreshes.next_due_at(settings, now(settings)) == []
