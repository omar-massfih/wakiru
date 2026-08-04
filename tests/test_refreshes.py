"""Refresh tests — the data jobs that ride each heartbeat wake.

Each job (mail snapshot, weather, ICS feeds, CalDAV) runs only when its interval
has elapsed since the last stamp, and re-stamps after every attempt so an outage
retries next interval rather than every wake. next_due_at feeds the wake
scheduler so a far-off self-paced wake can't starve them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from assistant import heartbeat, refreshes
from assistant.calendar.context import now
from assistant.calendar.store import parse_dt
from assistant.config import Settings
from assistant.trips import store as trips_store


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
    calls: list[datetime | None] = []
    monkeypatch.setattr(
        "assistant.weather.maybe_refresh",
        lambda s, instant=None: calls.append(instant),
    )

    ran = refreshes.run_due(settings)
    assert ran.get("weather") is True
    assert len(calls) == 1 and calls[0] is not None
    # The stamp is written, so the next immediate wake is a no-op.
    assert parse_dt(heartbeat.state_get(settings, "refresh:weather")) is not None


def test_run_due_skips_within_the_interval(settings, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "assistant.weather.maybe_refresh", lambda s, instant=None: calls.append("w")
    )

    refreshes.run_due(settings)  # first run stamps
    refreshes.run_due(settings)  # within 60 min: skipped
    assert calls == ["w"]


def test_failed_job_restamps_so_it_retries_next_interval(settings, monkeypatch) -> None:
    def boom(s, instant=None):
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
    monkeypatch.setattr("assistant.weather.maybe_refresh", lambda s, instant=None: None)
    refreshes.run_due(settings)
    current = now(settings)
    due = refreshes.next_due_at(settings, current)
    assert len(due) == 1
    # Roughly stamp + 60 min (the stamp was written ~now).
    assert abs(due[0] - (current + timedelta(minutes=60))) < timedelta(seconds=5)


def test_next_due_at_caps_recent_weather_refresh_at_trip_boundaries(settings) -> None:
    traveling = settings.model_copy(
        update={"enable_trips": True, "weather_refresh_minutes": 24 * 60}
    )
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-05", end="2026-08-05", timezone="UTC"
    )
    before_start = datetime(2026, 8, 4, 23, 55, tzinfo=UTC)
    heartbeat.state_set(traveling, "refresh:weather", before_start.isoformat())
    heartbeat.state_set(
        traveling,
        "refresh:weather:location",
        '["coords",59.9,10.7,""]',
    )
    assert refreshes.next_due_at(traveling, before_start) == [datetime(2026, 8, 5, tzinfo=UTC)]

    during_trip = datetime(2026, 8, 5, 23, 55, tzinfo=UTC)
    heartbeat.state_set(traveling, "refresh:weather", during_trip.isoformat())
    heartbeat.state_set(traveling, "refresh:weather:location", "name:tokyo")
    assert refreshes.next_due_at(traveling, during_trip) == [datetime(2026, 8, 6, tzinfo=UTC)]


def test_next_due_at_schedules_trip_start_without_a_home_location(tmp_path) -> None:
    traveling = Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="UTC",
        enable_weather=True,
        enable_trips=True,
        weather_refresh_minutes=24 * 60,
    )
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-05", end="2026-08-05", timezone="UTC"
    )
    before_start = datetime(2026, 8, 4, 23, 55, tzinfo=UTC)
    assert refreshes.next_due_at(traveling, before_start) == [datetime(2026, 8, 5, tzinfo=UTC)]


def test_disabled_jobs_are_neither_run_nor_scheduled(tmp_path, monkeypatch) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"))  # weather/email/feeds off
    monkeypatch.setattr(
        "assistant.weather.maybe_refresh",
        lambda s, instant=None: pytest.fail("weather is disabled"),
    )
    assert refreshes.run_due(settings) == {}
    assert refreshes.next_due_at(settings, now(settings)) == []


def test_weather_location_change_runs_immediately_and_is_due_now(settings, monkeypatch) -> None:
    from assistant import weather

    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    calls: list[str | None] = []
    monkeypatch.setattr(
        weather,
        "maybe_refresh",
        lambda s, instant=None: calls.append(weather.effective_location_key(s, instant)),
    )
    instant = datetime(2026, 8, 3, 23, 59, tzinfo=UTC)
    monkeypatch.setattr(refreshes, "now", lambda s: instant)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    refreshes.run_due(traveling)

    instant = datetime(2026, 8, 4, 0, tzinfo=UTC)
    assert refreshes.next_due_at(traveling, instant) == [instant]
    assert refreshes.run_due(traveling).get("weather") is True
    assert calls[0] != calls[1]


def test_failed_location_attempt_backs_off_then_return_home_attempts(settings, monkeypatch) -> None:
    from assistant import weather

    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    instant = datetime(2026, 8, 4, 12, tzinfo=UTC)
    monkeypatch.setattr(refreshes, "now", lambda s: instant)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    attempts = {"n": 0}

    def fail(s, instant=None):
        attempts["n"] += 1
        raise RuntimeError("forecast unavailable")

    monkeypatch.setattr(weather, "maybe_refresh", fail)
    assert refreshes.run_due(traveling).get("weather") is False
    assert heartbeat.state_get(traveling, "refresh:weather:location") == "name:tokyo"
    assert refreshes.run_due(traveling) == {}
    assert attempts["n"] == 1

    instant = datetime(2026, 8, 5, 0, tzinfo=UTC)
    assert refreshes.next_due_at(traveling, instant) == [instant]
    assert refreshes.run_due(traveling).get("weather") is False
    assert attempts["n"] == 2


def test_weather_worker_uses_scheduler_instant_across_trip_boundary(
    settings, monkeypatch
) -> None:
    from assistant import weather

    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    captured = datetime(2026, 8, 3, 23, 59, 59, tzinfo=UTC)
    after_boundary = datetime(2026, 8, 4, 0, tzinfo=UTC)
    monkeypatch.setattr(refreshes, "now", lambda s: captured)
    monkeypatch.setattr(weather, "now", lambda s: after_boundary)
    monkeypatch.setattr(
        weather,
        "_fetch",
        lambda *args, **kwargs: {
            "current": {"temperature_2m": 10, "weather_code": 0},
            "daily": {},
        },
    )

    assert refreshes.run_due(traveling).get("weather") is True
    stored = weather._load(traveling)
    assert stored is not None
    _, fetched_at, snapshot_location = stored
    attempted_location = heartbeat.state_get(traveling, "refresh:weather:location")
    assert fetched_at == captured
    assert snapshot_location == attempted_location
    assert snapshot_location.startswith('["coords"')
