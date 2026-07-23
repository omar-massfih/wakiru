"""Habit-log tests — store append/list, streak math, the summary rendering, and
the tool paths.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from assistant.config import Settings
from assistant.habits import store
from assistant.habits.context import current_streak, overview, summarize
from assistant.tools import ToolContext, tool_map


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_habits=True,
    )


# --- store + streaks -------------------------------------------------------- #


def test_log_and_list(settings) -> None:
    store.log_entry(settings, "sleep", value="7.5", unit="hours")
    entries = store.list_entries(settings)
    assert len(entries) == 1
    assert entries[0].habit == "sleep" and entries[0].value == 7.5


def test_list_filters_by_habit(settings) -> None:
    store.log_entry(settings, "gym")
    store.log_entry(settings, "sleep", value="8", unit="hours")
    assert [e.habit for e in store.list_entries(settings, "gym")] == ["gym"]


def test_current_streak_counts_consecutive_days(settings) -> None:
    today = date(2026, 7, 23)
    for offset in (0, 1, 2, 4):  # a gap at day 3 breaks the streak
        d = (today - timedelta(days=offset)).isoformat()
        store.log_entry(settings, "gym", on=d)
    entries = store.list_entries(settings, "gym")
    assert current_streak(entries, today) == 3  # today, -1, -2


def test_streak_broken_when_last_log_is_old(settings) -> None:
    today = date(2026, 7, 23)
    store.log_entry(settings, "gym", on=(today - timedelta(days=5)).isoformat())
    entries = store.list_entries(settings, "gym")
    assert current_streak(entries, today) == 0


def test_summary_and_overview_render(settings) -> None:
    store.log_entry(settings, "sleep", value="7", unit="hours")
    assert "sleep" in overview(settings, date.today())
    detail = summarize(settings, "sleep", date.today())
    assert "logged 1" in detail and "id:" in detail


# --- tools ------------------------------------------------------------------ #


def _run(settings, tool, **args) -> str:
    return tool_map(settings)[tool].run(ToolContext(settings=settings), **args)


def test_log_habit_tool(settings) -> None:
    out = _run(settings, "log_habit", habit="run", value="5", unit="km")
    assert "Logged run" in out and "km" in out
    assert len(store.list_entries(settings)) == 1


def test_habit_summary_and_remove_tools(settings) -> None:
    _run(settings, "log_habit", habit="weight", value="80", unit="kg")
    summary = _run(settings, "habit_summary", habit="weight")
    assert "weight" in summary
    entry = store.list_entries(settings, "weight")[0]
    assert "Removed" in _run(settings, "remove_habit_entry", id=entry.id)
    assert store.list_entries(settings) == []


def test_tools_absent_when_disabled(settings) -> None:
    off = settings.model_copy(update={"enable_habits": False})
    assert "log_habit" not in tool_map(off)
