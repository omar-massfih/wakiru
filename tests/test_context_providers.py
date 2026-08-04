"""Context-provider registry tests — gating, isolation, and ordering."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from assistant import weather
from assistant.config import Settings
from assistant.context_providers import ContextProvider, build_context
from assistant.trips import store as trips_store


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), timezone="Europe/Oslo")


def _provider(name: str, enabled: bool, text: str) -> ContextProvider:
    return ContextProvider(name, lambda s, e=enabled: e, lambda ctx, t=text: t)


def test_disabled_provider_is_omitted_entirely(settings) -> None:
    blocks = build_context(
        settings,
        "q",
        "t1",
        providers=[_provider("on", True, "hello"), _provider("off", False, "hidden")],
    )
    assert blocks == {"on": "hello"}


def test_failing_provider_contributes_empty_and_never_starves_others(settings) -> None:
    def boom(ctx):
        raise RuntimeError("subsystem down")

    blocks = build_context(
        settings,
        "q",
        "t1",
        providers=[
            ContextProvider("broken", lambda s: True, boom),
            _provider("healthy", True, "still here"),
        ],
    )
    assert blocks == {"broken": "", "healthy": "still here"}


def test_registry_order_is_block_order(settings) -> None:
    blocks = build_context(
        settings,
        "q",
        "t1",
        providers=[_provider(n, True, n) for n in ("b", "a", "c")],
    )
    assert list(blocks) == ["b", "a", "c"]


def test_provider_sees_the_turn(settings) -> None:
    seen = {}

    def capture(ctx):
        seen["query"] = ctx.query
        seen["thread_id"] = ctx.thread_id
        return ""

    build_context(
        settings, "find my notes", "telegram:7",
        providers=[ContextProvider("cap", lambda s: True, capture)],
    )
    assert seen == {"query": "find my notes", "thread_id": "telegram:7"}


def test_default_registry_gates_follow_settings(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )
    blocks = build_context(settings, "q", "t1")
    # Email is off by default; the always-on features contribute blocks.
    assert "mail" not in blocks
    assert {"recall", "profile", "agenda", "tasks"} <= set(blocks)
    assert "Current date and time" in blocks["agenda"]

    lean = settings.model_copy(update={"enable_calendar": False, "enable_tasks": False})
    lean_blocks = build_context(lean, "q", "t1")
    assert "agenda" not in lean_blocks and "tasks" not in lean_blocks


def test_weather_context_tracks_trip_and_withholds_failed_transition(
    settings, monkeypatch
) -> None:
    traveling = settings.model_copy(
        update={
            "enable_weather": True,
            "enable_trips": True,
            "weather_latitude": 59.9,
            "weather_longitude": 10.7,
            "weather_location_name": "Oslo",
            "weather_refresh_minutes": 60,
        }
    )
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    payload = {
        "current": {"temperature_2m": 20, "weather_code": 0},
        "daily": {
            "temperature_2m_max": [24],
            "temperature_2m_min": [16],
            "weather_code": [0],
        },
    }
    instant = datetime(2026, 8, 4, 12, tzinfo=UTC)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    monkeypatch.setattr(weather, "_geocode", lambda s, name: (35.68, 139.65))
    monkeypatch.setattr(weather, "_fetch", lambda *args, **kwargs: payload)
    weather.refresh(traveling)
    assert "Location: Tokyo" in build_context(traveling, "q", "t1")["weather"]

    instant = datetime(2026, 8, 5, 0, tzinfo=UTC)
    monkeypatch.setattr(
        weather, "_fetch", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    weather.refresh(traveling)
    assert build_context(traveling, "q", "t1")["weather"] == ""

    monkeypatch.setattr(weather, "_fetch", lambda *args, **kwargs: payload)
    weather.refresh(traveling)
    assert "Location: Oslo" in build_context(traveling, "q", "t1")["weather"]
