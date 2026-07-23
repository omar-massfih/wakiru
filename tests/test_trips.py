"""Trip tests — store CRUD, the context block, and the tool paths.

Everything runs for real (plain SQLite); the tools are exercised through
``tool_map`` exactly as the agent dispatches them. Dates are pinned relative
to the assistant's own "today" so the tests hold on any calendar day.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant.calendar.context import now
from assistant.config import Settings
from assistant.tools import ToolContext, available_tools, tool_map
from assistant.trips import store, trips_context


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_trips=True,
    )


def _day(settings: Settings, offset: int) -> str:
    return (now(settings).date() + timedelta(days=offset)).isoformat()


# --- store CRUD ------------------------------------------------------------- #


def test_create_list_orders_soonest_first(settings) -> None:
    store.create_trip(settings, "Lisbon", start=_day(settings, 30), end=_day(settings, 37))
    store.create_trip(settings, "Bergen", start=_day(settings, 5), end=_day(settings, 7))
    assert [t.destination for t in store.list_trips(settings)] == ["Bergen", "Lisbon"]


def test_past_trips_hidden_unless_asked(settings) -> None:
    store.create_trip(settings, "Rome", start=_day(settings, -20), end=_day(settings, -14))
    store.create_trip(settings, "Oslo", start=_day(settings, 3), end=_day(settings, 4))
    assert [t.destination for t in store.list_trips(settings)] == ["Oslo"]
    both = store.list_trips(settings, include_past=True)
    assert [t.destination for t in both] == ["Oslo", "Rome"]


def test_active_and_next_trip(settings) -> None:
    store.create_trip(settings, "Lisbon", start=_day(settings, -2), end=_day(settings, 3))
    store.create_trip(settings, "Bergen", start=_day(settings, 10), end=_day(settings, 12))
    assert store.active_trip(settings).destination == "Lisbon"
    assert store.next_trip(settings).destination == "Bergen"


def test_find_prefers_live_trips_over_past(settings) -> None:
    old = store.create_trip(settings, "Lisbon", start=_day(settings, -30), end=_day(settings, -25))
    new = store.create_trip(settings, "Lisbon", start=_day(settings, 20), end=_day(settings, 27))
    assert store.find_trip(settings, "lisbon").id == new.id
    assert store.find_trip(settings, old.id).id == old.id


# --- context block ---------------------------------------------------------- #


def test_context_empty_without_travel(settings) -> None:
    assert trips_context(settings) == ""
    # A trip too far out stays silent too.
    store.create_trip(settings, "Tokyo", start=_day(settings, 60), end=_day(settings, 74))
    assert trips_context(settings) == ""


def test_context_surfaces_imminent_departure(settings) -> None:
    store.create_trip(
        settings, "Lisbon", start=_day(settings, 4), end=_day(settings, 9),
        notes="TP753 out of OSL",
    )
    block = trips_context(settings)
    assert "Upcoming trip" in block and "Lisbon" in block
    assert "departs in 4 day(s)" in block
    assert "TP753 out of OSL" in block


def test_context_surfaces_active_trip_with_local_time(settings) -> None:
    store.create_trip(
        settings, "Lisbon", start=_day(settings, -1), end=_day(settings, 5),
        timezone="Europe/Lisbon",
    )
    block = trips_context(settings)
    assert "Trip in progress" in block
    assert "day 2 of 7" in block
    assert "Local time in Lisbon" in block
    assert "get_weather" in block


def test_context_mentions_packing_list_only_with_lists(settings) -> None:
    store.create_trip(settings, "Bergen", start=_day(settings, 2), end=_day(settings, 3))
    assert "add_to_list" not in trips_context(settings)
    with_lists = settings.model_copy(update={"enable_lists": True})
    assert "add_to_list" in trips_context(with_lists)


# --- tool path -------------------------------------------------------------- #


def _run(settings, name, **args) -> str:
    return tool_map(settings)[name].run(ToolContext(settings=settings), **args)


def test_add_and_list_tools(settings) -> None:
    out = _run(
        settings, "add_trip", destination="Lisbon",
        start=_day(settings, 10), end=_day(settings, 17), timezone="Europe/Lisbon",
    )
    assert out.startswith("Trip saved: Lisbon")
    listing = _run(settings, "list_trips")
    assert "Lisbon" in listing and "Europe/Lisbon" in listing


def test_add_tool_validates_dates_and_timezone(settings) -> None:
    assert "not a YYYY-MM-DD date" in _run(
        settings, "add_trip", destination="X", start="next friday"
    )
    assert "ends before it starts" in _run(
        settings, "add_trip", destination="X",
        start=_day(settings, 5), end=_day(settings, 2),
    )
    assert "not an IANA timezone" in _run(
        settings, "add_trip", destination="X", timezone="Lisbon time"
    )
    assert store.list_trips(settings, include_past=True) == []


def test_update_and_remove_tools(settings) -> None:
    store.create_trip(settings, "Bergen", start=_day(settings, 5), end=_day(settings, 6))
    out = _run(settings, "update_trip", trip="bergen", end=_day(settings, 8))
    assert "Trip updated" in out and _day(settings, 8) in out
    assert "ends before it starts" in _run(
        settings, "update_trip", trip="bergen", end=_day(settings, 1)
    )
    assert "Trip removed: Bergen" in _run(settings, "remove_trip", trip="bergen")
    assert store.list_trips(settings, include_past=True) == []


def test_tools_absent_when_disabled_and_writes_chat_only(settings) -> None:
    off = settings.model_copy(update={"enable_trips": False})
    assert "add_trip" not in tool_map(off)
    s = settings.model_copy(update={"enable_heartbeat": True})
    beat = {spec.name for spec in available_tools(s, mode="heartbeat")}
    assert not {"add_trip", "update_trip", "remove_trip"} & beat
    assert "list_trips" in beat
