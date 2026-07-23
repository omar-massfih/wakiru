"""Checklist tests — store CRUD, the tool paths, and the resolve helpers.

Everything runs for real (plain SQLite); the tools are exercised through
``tool_map`` exactly as the agent dispatches them.
"""

from __future__ import annotations

import pytest

from assistant.config import Settings
from assistant.lists import store
from assistant.tools import ToolContext, available_tools, tool_map


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_lists=True,
    )


# --- store CRUD ------------------------------------------------------------- #


def test_add_and_list_keeps_insertion_order(settings) -> None:
    store.add_item(settings, "shopping", "milk")
    store.add_item(settings, "shopping", "eggs")
    items = store.list_items(settings, "shopping")
    assert [e.item for e in items] == ["milk", "eggs"]
    assert all(not e.done for e in items)


def test_list_name_matching_is_case_insensitive(settings) -> None:
    store.add_item(settings, "Shopping", "milk")
    store.add_item(settings, "shopping", "eggs")
    assert store.list_names(settings) == [("Shopping", 2)]
    assert [e.item for e in store.list_items(settings, "SHOPPING")] == ["milk", "eggs"]


def test_check_off_moves_out_of_open(settings) -> None:
    entry = store.add_item(settings, "errands", "post office")
    store.set_done(settings, entry.id)
    assert store.list_items(settings, "errands") == []
    everything = store.list_items(settings, "errands", include_done=True)
    assert len(everything) == 1 and everything[0].done is True
    assert everything[0].done_at != ""
    # The emptied list still shows up (open count 0) until entries are removed.
    assert store.list_names(settings) == [("errands", 0)]


def test_find_by_id_and_text_scoped_to_list(settings) -> None:
    milk = store.add_item(settings, "shopping", "milk")
    store.add_item(settings, "packing", "milk powder")
    assert store.find_item(settings, milk.id).id == milk.id
    assert store.find_item(settings, "milk", "shopping").id == milk.id
    assert store.find_item(settings, "milk", "packing").item == "milk powder"
    assert store.find_item(settings, "nothing") is None


def test_open_items_shadow_done_in_find(settings) -> None:
    done = store.add_item(settings, "shopping", "bread")
    store.set_done(settings, done.id)
    fresh = store.add_item(settings, "shopping", "bread")
    assert store.find_item(settings, "bread").id == fresh.id


# --- tool path -------------------------------------------------------------- #


def _run(settings, name, **args) -> str:
    return tool_map(settings)[name].run(ToolContext(settings=settings), **args)


def test_add_tool_splits_multiple_items(settings) -> None:
    out = _run(settings, "add_to_list", list="shopping", items="milk, eggs\nbread")
    assert "Added to the shopping list: milk, eggs, bread" in out
    assert [e.item for e in store.list_items(settings, "shopping")] == [
        "milk",
        "eggs",
        "bread",
    ]


def test_show_list_tool_renders_one_or_all(settings) -> None:
    _run(settings, "add_to_list", list="shopping", items="milk")
    _run(settings, "add_to_list", list="packing", items="passport")
    one = _run(settings, "show_list", list="shopping")
    assert "milk" in one and "passport" not in one
    both = _run(settings, "show_list")
    assert "milk" in both and "passport" in both
    assert "There are no lists yet." in _run(
        settings.model_copy(update={"memory_dir": settings.memory_dir + "2"}),
        "show_list",
    )


def test_check_off_and_remove_tools(settings) -> None:
    _run(settings, "add_to_list", list="shopping", items="milk")
    assert "Checked off “milk”" in _run(settings, "check_off_item", item="milk")
    assert store.list_items(settings, "shopping") == []
    assert "Removed “milk”" in _run(settings, "remove_from_list", item="milk")
    assert store.list_items(settings, "shopping", include_done=True) == []


def test_tools_absent_when_disabled(settings) -> None:
    off = settings.model_copy(update={"enable_lists": False})
    assert "add_to_list" not in tool_map(off)


def test_list_writes_are_chat_only(settings) -> None:
    s = settings.model_copy(update={"enable_heartbeat": True})
    beat = {spec.name for spec in available_tools(s, mode="heartbeat")}
    assert not {"add_to_list", "check_off_item", "remove_from_list"} & beat
    assert "show_list" in beat  # read-only: briefing enrichment is fine
    chat = {spec.name for spec in available_tools(s)}
    assert {"add_to_list", "check_off_item", "remove_from_list", "show_list"} <= chat
