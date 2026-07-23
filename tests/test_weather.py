"""Weather snapshot tests — rendering, the cache round-trip, and the freshness
and configuration gates. The network is never touched: ``_fetch`` is monkeypatched
to return a canned Open-Meteo payload, matching how the mail-snapshot tests fake IMAP.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant import weather
from assistant.calendar.context import now
from assistant.config import Settings

_PAYLOAD = {
    "current": {
        "temperature_2m": 9.0,
        "apparent_temperature": 7.0,
        "precipitation": 0.2,
        "weather_code": 61,
        "wind_speed_10m": 12.0,
    },
    "daily": {
        "temperature_2m_max": [11.0],
        "temperature_2m_min": [4.0],
        "precipitation_probability_max": [60],
        "weather_code": [3],
    },
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_weather=True,
        weather_latitude=59.9,
        weather_longitude=10.7,
        weather_location_name="Oslo",
        weather_refresh_minutes=60,
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch) -> None:
    monkeypatch.setattr(weather, "_fetch", lambda s, lat, lon: _PAYLOAD)


def test_render_covers_now_and_today(settings) -> None:
    text = weather._render(settings, _PAYLOAD)
    assert "Oslo" in text
    assert "9.0°C" in text
    assert "feels 7.0°C" in text
    assert "light rain" in text  # weather_code 61
    assert "4.0–11.0°C" in text
    assert "overcast" in text  # daily weather_code 3
    assert "60% chance of precipitation" in text


def test_imperial_units(settings) -> None:
    settings = settings.model_copy(update={"weather_units": "imperial"})
    text = weather._render(settings, _PAYLOAD)
    assert "°F" in text and "mph" in text


def test_refresh_then_current_returns_block(settings) -> None:
    assert weather.refresh(settings) is not None
    block = weather.current(settings)
    assert block.startswith("## Weather (as of ")
    assert "light rain" in block


def test_disabled_is_a_no_op(settings) -> None:
    settings = settings.model_copy(update={"enable_weather": False})
    assert weather.refresh(settings) is None
    assert weather.current(settings) == ""


def test_no_location_disables(settings) -> None:
    # Neither coordinates nor a place name — nothing to forecast for.
    settings = settings.model_copy(
        update={
            "weather_latitude": None,
            "weather_longitude": None,
            "weather_location_name": "",
        }
    )
    assert weather.enabled(settings) is False
    assert weather.refresh(settings) is None


def test_geocodes_place_name_when_no_coords(settings, monkeypatch) -> None:
    settings = settings.model_copy(
        update={
            "weather_latitude": None,
            "weather_longitude": None,
            "weather_location_name": "Oslo",
        }
    )
    # A place name alone is a location (resolved off the reply path).
    assert weather.enabled(settings) is True
    calls = {"n": 0}

    def _geo(s, name):
        calls["n"] += 1
        assert name == "Oslo"
        return (59.91, 10.75)

    monkeypatch.setattr(weather, "_geocode", _geo)
    assert weather.refresh(settings) is not None
    assert "light rain" in weather.current(settings)
    # A second refresh reuses the cached coordinates — no re-geocode.
    weather.refresh(settings)
    assert calls["n"] == 1


def test_explicit_coords_skip_geocoding(settings, monkeypatch) -> None:
    def _boom(s, name):
        raise AssertionError("should not geocode when coordinates are configured")

    monkeypatch.setattr(weather, "_geocode", _boom)
    assert weather.refresh(settings) is not None  # fixture has lat/lon set


def test_stale_snapshot_withheld(settings, monkeypatch) -> None:
    weather.refresh(settings)
    later = now(settings) + timedelta(hours=weather._MAX_AGE_HOURS + 1)
    monkeypatch.setattr(weather, "now", lambda s: later)
    assert weather.current(settings) == ""


def test_maybe_refresh_respects_cadence(settings, monkeypatch) -> None:
    calls = {"n": 0}

    def _counting_fetch(s, lat, lon):
        calls["n"] += 1
        return _PAYLOAD

    monkeypatch.setattr(weather, "_fetch", _counting_fetch)
    weather.maybe_refresh(settings)  # first: fetches
    weather.maybe_refresh(settings)  # still fresh: no fetch
    assert calls["n"] == 1
    # Past the cadence, it refetches.
    later = now(settings) + timedelta(minutes=settings.weather_refresh_minutes + 1)
    monkeypatch.setattr(weather, "now", lambda s: later)
    weather.maybe_refresh(settings)
    assert calls["n"] == 2
