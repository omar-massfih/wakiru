"""Weather snapshot tests — rendering, the cache round-trip, and the freshness
and configuration gates. The network is never touched: ``_fetch`` is monkeypatched
to return a canned Open-Meteo payload, matching how the mail-snapshot tests fake IMAP.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from assistant import weather
from assistant.calendar.context import now
from assistant.config import Settings
from assistant.trips import store as trips_store

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
    monkeypatch.setattr(weather, "_fetch", lambda s, lat, lon, days=1: _PAYLOAD)


def test_render_covers_now_and_today(settings) -> None:
    text = weather._render(settings, _PAYLOAD)
    assert "Oslo" in text
    assert "9.0°C" in text
    assert "feels 7.0°C" in text
    assert "light rain" in text  # weather_code 61
    assert "4.0–11.0°C" in text
    assert "overcast" in text  # daily weather_code 3
    assert "60% chance of precipitation" in text


def test_severe_conditions_get_a_heads_up(settings) -> None:
    severe = {
        "current": _PAYLOAD["current"],
        "daily": {
            "temperature_2m_max": [-2.0],
            "temperature_2m_min": [-8.0],
            "precipitation_probability_max": [90],
            "weather_code": [75],  # heavy snow
        },
    }
    text = weather._render(settings, severe)
    assert "⚠" in text and "snow" in text.lower()
    # A benign day carries no alert.
    assert "⚠" not in weather._render(settings, _PAYLOAD)  # code 3 = overcast


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


_MULTI_DAY = {
    "current": _PAYLOAD["current"],
    "daily": {
        "time": ["2026-07-23", "2026-07-24", "2026-07-25"],
        "temperature_2m_max": [11.0, 13.0, 10.0],
        "temperature_2m_min": [4.0, 6.0, 5.0],
        "precipitation_probability_max": [60, 20, 80],
        "weather_code": [3, 1, 61],
    },
}


def test_forecast_for_geocodes_and_renders_multi_day(settings, monkeypatch) -> None:
    monkeypatch.setattr(weather, "_geocode", lambda s, name: (60.39, 5.32))
    monkeypatch.setattr(weather, "_fetch", lambda s, lat, lon, days=1: _MULTI_DAY)
    out = weather.forecast_for(settings, "Bergen", days=3)
    assert out.startswith("Weather for Bergen:")
    assert "Now: 9.0°C" in out
    weekday = datetime.strptime("2026-07-24", "%Y-%m-%d").strftime("%a")
    assert f"{weekday} 2026-07-24: 6.0–13.0°C" in out  # weekday label rendered
    assert "80% precip" in out


def test_forecast_for_unresolved_location(settings, monkeypatch) -> None:
    monkeypatch.setattr(weather, "_geocode", lambda s, name: None)
    assert weather.forecast_for(settings, "Nowheresville") is None


def test_get_weather_tool(settings, monkeypatch) -> None:
    from assistant.tools import ToolContext, tool_map

    settings = settings.model_copy(update={"enable_weather": True})
    monkeypatch.setattr(weather, "_geocode", lambda s, name: (60.39, 5.32))
    monkeypatch.setattr(weather, "_fetch", lambda s, lat, lon, days=1: _MULTI_DAY)
    spec = tool_map(settings)["get_weather"]
    result = spec.run(ToolContext(settings=settings), location="Bergen", days="2")
    assert "Weather for Bergen:" in result


def test_get_weather_absent_when_weather_off(settings) -> None:
    from assistant.tools import tool_map

    off = settings.model_copy(update={"enable_weather": False})
    assert "get_weather" not in tool_map(off)


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


def test_trip_destination_follows_destination_local_inclusive_boundaries(
    settings, monkeypatch
) -> None:
    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling,
        "New York",
        start="2026-08-10",
        end="2026-08-10",
        timezone="America/New_York",
    )
    geocodes: list[str] = []
    fetches: list[tuple[float, float]] = []
    monkeypatch.setattr(
        weather,
        "_geocode",
        lambda s, name: geocodes.append(name) or (40.71, -74.0),
    )
    monkeypatch.setattr(
        weather,
        "_fetch",
        lambda s, lat, lon, days=1: fetches.append((lat, lon)) or _PAYLOAD,
    )

    instant = datetime(2026, 8, 10, 3, 59, 59, tzinfo=UTC)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    weather.refresh(traveling)
    assert fetches[-1] == (59.9, 10.7)
    assert "Location: Oslo" in weather.current(traveling)

    instant = datetime(2026, 8, 10, 4, 0, tzinfo=UTC)
    weather.maybe_refresh(traveling)  # location mismatch bypasses the cadence
    assert fetches[-1] == (40.71, -74.0)
    assert "Location: New York" in weather.current(traveling)

    instant = datetime(2026, 8, 11, 3, 59, 59, tzinfo=UTC)
    assert weather.effective_location_key(traveling) == "name:new york"
    instant = datetime(2026, 8, 11, 4, 0, tzinfo=UTC)
    weather.maybe_refresh(traveling)
    assert fetches[-1] == (59.9, 10.7)
    assert "Location: Oslo" in weather.current(traveling)
    assert geocodes == ["New York"]


@pytest.mark.parametrize("trip_timezone", ["", "Not/AZone"])
def test_timezone_less_or_corrupt_trip_uses_home_boundaries(
    settings, monkeypatch, trip_timezone
) -> None:
    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling,
        "Bergen",
        start="2026-08-10",
        end="2026-08-10",
        timezone=trip_timezone,
    )
    monkeypatch.setattr(weather, "_geocode", lambda s, name: (60.39, 5.32))
    instant = datetime(2026, 8, 9, 21, 59, 59, tzinfo=UTC)  # 23:59:59 in Oslo
    monkeypatch.setattr(weather, "now", lambda s: instant)
    assert weather.effective_location_key(traveling).startswith("[\"coords\"")
    instant = datetime(2026, 8, 9, 22, 0, tzinfo=UTC)  # home-local midnight
    assert weather.effective_location_key(traveling) == "name:bergen"


def test_trip_can_supply_location_without_home_and_trips_can_be_disabled(
    settings, monkeypatch
) -> None:
    no_home = settings.model_copy(
        update={
            "enable_trips": True,
            "weather_latitude": None,
            "weather_longitude": None,
            "weather_location_name": "",
        }
    )
    trips_store.create_trip(
        no_home, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    instant = datetime(2026, 8, 4, 12, tzinfo=UTC)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    monkeypatch.setattr(weather, "_geocode", lambda s, name: (35.68, 139.65))
    assert weather.refresh(no_home) is not None
    assert "Location: Tokyo" in weather.current(no_home)
    assert weather.enabled(no_home.model_copy(update={"enable_trips": False})) is False


def test_named_geocode_cache_reuses_home_trip_and_normalized_destinations(
    settings, monkeypatch
) -> None:
    traveling = settings.model_copy(
        update={
            "enable_trips": True,
            "weather_latitude": None,
            "weather_longitude": None,
        }
    )
    trips_store.create_trip(
        traveling, " New   York ", start="2026-08-10", end="2026-08-10", timezone="UTC"
    )
    calls: list[str] = []

    def geocode(s, name):
        calls.append(name)
        return (40.7, -74.0) if "York" in name else (59.9, 10.7)

    monkeypatch.setattr(weather, "_geocode", geocode)
    instant = datetime(2026, 8, 9, 12, tzinfo=UTC)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    weather.refresh(traveling)
    instant = datetime(2026, 8, 10, 12, tzinfo=UTC)
    weather.refresh(traveling)
    weather.refresh(traveling)
    instant = datetime(2026, 8, 11, 12, tzinfo=UTC)
    weather.refresh(traveling)
    assert calls == ["Oslo", "New   York"]


def test_legacy_geocode_cache_is_read_and_upgraded(settings, monkeypatch) -> None:
    named = settings.model_copy(
        update={"weather_latitude": None, "weather_longitude": None}
    )
    named.memory_path.mkdir(parents=True)
    (named.memory_path / "weather_geocode.json").write_text(
        json.dumps({"name": "OSLO", "latitude": 59.91, "longitude": 10.75}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        weather, "_geocode", lambda s, name: pytest.fail("legacy entry should be reused")
    )
    assert weather.refresh(named) is not None


def test_location_change_withholds_snapshot_and_failure_preserves_it(
    settings, monkeypatch
) -> None:
    traveling = settings.model_copy(update={"enable_trips": True})
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    instant = datetime(2026, 8, 3, 23, 59, tzinfo=UTC)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    weather.refresh(traveling)
    saved = weather._path(traveling).read_text(encoding="utf-8")

    instant = datetime(2026, 8, 4, 0, tzinfo=UTC)
    assert weather.current(traveling) == ""
    monkeypatch.setattr(weather, "_geocode", lambda s, name: None)
    assert weather.refresh(traveling) is None
    assert weather.current(traveling) == ""
    assert weather._path(traveling).read_text(encoding="utf-8") == saved


def test_failed_home_restore_does_not_expose_trip_snapshot(settings, monkeypatch) -> None:
    traveling = settings.model_copy(
        update={
            "enable_trips": True,
            "weather_latitude": None,
            "weather_longitude": None,
        }
    )
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    monkeypatch.setattr(weather, "_geocode", lambda s, name: (35.68, 139.65))
    instant = datetime(2026, 8, 4, 12, tzinfo=UTC)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    weather.refresh(traveling)
    assert "Tokyo" in weather.current(traveling)

    instant = datetime(2026, 8, 5, 0, tzinfo=UTC)
    monkeypatch.setattr(weather, "_geocode", lambda s, name: None)
    assert weather.refresh(traveling) is None
    assert weather.current(traveling) == ""


def test_legacy_snapshot_is_unverified_and_location_inputs_invalidate(settings) -> None:
    captured = now(settings)
    settings.memory_path.mkdir(parents=True)
    weather._path(settings).write_text(
        json.dumps({"text": "old", "fetched_at": captured.isoformat()}), encoding="utf-8"
    )
    assert weather.current(settings) == ""

    weather.refresh(settings)
    moved = settings.model_copy(update={"weather_latitude": 60.0})
    relabeled = settings.model_copy(update={"weather_location_name": "Oslo home"})
    assert weather.current(moved) == ""
    assert weather.current(relabeled) == ""


def test_same_location_failed_refresh_keeps_fresh_honest_snapshot(
    settings, monkeypatch
) -> None:
    weather.refresh(settings)
    monkeypatch.setattr(
        weather, "_fetch", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("down"))
    )
    assert weather.refresh(settings) is None
    assert "Location: Oslo" in weather.current(settings)
