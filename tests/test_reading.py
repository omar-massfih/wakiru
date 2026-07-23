"""Reading-list tests — store CRUD, the tool paths, and the resolve helpers.

Everything runs for real (plain SQLite); the tools are exercised through
``tool_map`` exactly as the agent dispatches them.
"""

from __future__ import annotations

import pytest

from assistant.config import Settings
from assistant.reading import store
from assistant.tools import ToolContext, tool_map


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_reading=True,
    )


# --- store CRUD ------------------------------------------------------------- #


def test_create_and_list_unread(settings) -> None:
    store.create_item(settings, "https://example.com/a", title="Article A")
    items = store.list_items(settings)
    assert [i.title for i in items] == ["Article A"]
    assert items[0].read is False


def test_title_defaults_to_url(settings) -> None:
    item = store.create_item(settings, "https://example.com/x")
    assert item.title == "https://example.com/x"


def test_mark_read_moves_out_of_unread(settings) -> None:
    item = store.create_item(settings, "https://example.com/a", title="A")
    store.mark_read(settings, item.id)
    assert store.list_items(settings) == []  # unread only
    allitems = store.list_items(settings, include_read=True)
    assert len(allitems) == 1 and allitems[0].read is True
    assert allitems[0].read_at != ""


def test_find_by_id_url_and_title(settings) -> None:
    item = store.create_item(settings, "https://example.com/rust", title="Learn Rust")
    assert store.find_item(settings, item.id).id == item.id
    assert store.find_item(settings, "rust").id == item.id  # url substring
    assert store.find_item(settings, "learn").id == item.id  # title substring
    assert store.find_item(settings, "nothing") is None


# --- tool path -------------------------------------------------------------- #


def _run(settings, name, **args) -> str:
    return tool_map(settings)[name].run(ToolContext(settings=settings), **args)


def test_save_and_list_tools(settings) -> None:
    out = _run(settings, "save_reading", url="https://example.com/a", title="A", note="for work")
    assert "Saved to your reading list: A" in out
    listing = _run(settings, "list_reading")
    assert "A" in listing and "for work" in listing


def test_save_rejects_non_http(settings) -> None:
    out = _run(settings, "save_reading", url="ftp://nope")
    assert "not an http(s) URL" in out
    assert store.list_items(settings) == []


def test_mark_read_and_remove_tools(settings) -> None:
    store.create_item(settings, "https://example.com/a", title="Alpha")
    assert "Marked read: Alpha" in _run(settings, "mark_read", query="alpha")
    assert store.list_items(settings) == []  # no longer unread
    assert "Removed from reading list: Alpha" in _run(settings, "remove_reading", query="alpha")
    assert store.list_items(settings, include_read=True) == []


def test_tools_absent_when_disabled(settings) -> None:
    off = settings.model_copy(update={"enable_reading": False})
    assert "save_reading" not in tool_map(off)
