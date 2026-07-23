"""Work-log tests — the timer lifecycle, direct logs, rollups, and the tools.

No LLM: the store and read paths run for real against tmp SQLite files; the
tool layer is exercised through tool_map like the other subsystem tests.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from assistant.calendar.context import resolve_tz
from assistant.config import Settings
from assistant.tools import ToolContext, tool_map
from assistant.worklog import store
from assistant.worklog.context import (
    fmt_minutes,
    summary,
    timer_context,
    totals_by_project,
    weekly_section,
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_worklog=True,
    )


def _freeze(monkeypatch, settings: Settings, at: datetime) -> None:
    frozen = at.replace(tzinfo=resolve_tz(settings))
    monkeypatch.setattr("assistant.calendar.context.now", lambda s: frozen)
    monkeypatch.setattr(
        store, "_stamp_now", lambda s: frozen.isoformat(timespec="seconds")
    )


# --- store: timer lifecycle ------------------------------------------------- #


def test_start_and_stop_records_duration(settings, monkeypatch) -> None:
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 9, 0))
    started, stopped = store.start_entry(settings, "wakiru", note="review")
    assert stopped is None
    assert store.running_entry(settings).id == started.id

    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 10, 30))
    entry = store.stop_entry(settings)
    assert entry is not None and entry.minutes == 90
    assert entry.worked_on == "2026-07-20"
    assert store.running_entry(settings) is None


def test_starting_again_stops_the_running_timer(settings, monkeypatch) -> None:
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 9, 0))
    store.start_entry(settings, "wakiru")
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 9, 45))
    started, stopped = store.start_entry(settings, "client X")
    assert stopped is not None and stopped.project == "wakiru" and stopped.minutes == 45
    assert store.running_entry(settings).project == "client X"
    assert started.worked_on == "2026-07-20"


def test_stop_without_running_timer(settings) -> None:
    assert store.stop_entry(settings) is None


def test_immediate_stop_floors_at_one_minute(settings, monkeypatch) -> None:
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 9, 0))
    store.start_entry(settings, "wakiru")
    entry = store.stop_entry(settings)
    assert entry.minutes == 1


def test_direct_log_and_validation(settings, monkeypatch) -> None:
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 18, 0))
    entry = store.log_entry(settings, "budget", "120", on="2026-07-19")
    assert entry.minutes == 120 and entry.worked_on == "2026-07-19"
    assert store.log_entry(settings, "", "60") is None
    assert store.log_entry(settings, "x", "0") is None
    assert store.log_entry(settings, "x", "junk") is None


def test_list_filters_case_insensitively(settings) -> None:
    store.log_entry(settings, "Wakiru", "30")
    store.log_entry(settings, "client X", "60")
    assert [e.project for e in store.list_entries(settings, "wakiru")] == ["Wakiru"]
    assert store.project_names(settings) == ["client X", "Wakiru"] or set(
        store.project_names(settings)
    ) == {"client X", "Wakiru"}


def test_delete_entry(settings) -> None:
    entry = store.log_entry(settings, "wakiru", "30")
    assert store.delete_entry(settings, entry.id).id == entry.id
    assert store.delete_entry(settings, entry.id) is None
    assert store.list_entries(settings) == []


# --- read paths -------------------------------------------------------------- #


def test_fmt_minutes() -> None:
    assert fmt_minutes(45) == "45m"
    assert fmt_minutes(120) == "2h"
    assert fmt_minutes(125) == "2h 05m"


def test_totals_group_case_insensitively(settings) -> None:
    store.log_entry(settings, "wakiru", "30", on="2026-07-19")
    store.log_entry(settings, "Wakiru", "60", on="2026-07-20")
    totals = totals_by_project(store.list_entries(settings))
    assert list(totals.values()) == [90]


def test_timer_context_empty_and_running(settings, monkeypatch) -> None:
    assert timer_context(settings) == ""
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 9, 0))
    store.start_entry(settings, "wakiru")
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 9, 20))
    block = timer_context(settings)
    assert "wakiru" in block and "20m" in block and "stop_work" in block


def test_summary_covers_today_and_window(settings, monkeypatch) -> None:
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 18, 0))
    today = date(2026, 7, 20)
    store.log_entry(settings, "wakiru", "60")
    store.log_entry(settings, "client X", "120", on=(today - timedelta(days=2)).isoformat())
    store.log_entry(settings, "old", "600", on=(today - timedelta(days=30)).isoformat())
    text = summary(settings, today)
    assert "Today: 1h (wakiru 1h)" in text
    assert "Last 7 days: 3h" in text
    assert "client X: 2h" in text
    assert "old" not in text.split("Recent entries:")[0]


def test_weekly_section(settings, monkeypatch) -> None:
    today = date(2026, 7, 20)
    assert weekly_section(settings, today) == ""
    store.log_entry(settings, "wakiru", "90", on="2026-07-18")
    text = weekly_section(settings, today)
    assert "Time worked last 7 days" in text and "1h 30m" in text


def test_weekly_review_carries_worklog(settings, monkeypatch) -> None:
    from assistant import weekly_review

    settings.enable_weekly_review = True
    store.log_entry(settings, "wakiru", "90", on="2026-07-10")
    frozen = datetime(2026, 7, 12, 18, 0, tzinfo=resolve_tz(settings))  # Sunday
    monkeypatch.setattr(weekly_review, "now", lambda s: frozen)
    monkeypatch.setattr(
        "assistant.compose.compose_push", lambda s, **kw: kw["fallback"]
    )
    sent: list[dict] = []
    monkeypatch.setattr(
        weekly_review, "deliver_reminder", lambda s, r, **kw: sent.append(r) or True
    )
    assert weekly_review.run_weekly_review(settings)["sent"]
    assert "Time worked last 7 days" in sent[0]["message"]


# --- tools ------------------------------------------------------------------- #


def _ctx(settings: Settings) -> ToolContext:
    return ToolContext(settings=settings, thread_id="t", batch_id="b")


def test_tools_registered_only_when_enabled(settings) -> None:
    names = set(tool_map(settings))
    assert {"start_work", "stop_work", "log_work", "work_summary", "remove_work_entry"} <= names
    settings.enable_worklog = False
    assert "start_work" not in tool_map(settings)


def test_heartbeat_mode_keeps_only_the_summary(settings) -> None:
    from assistant.tools import available_tools

    names = {t.name for t in available_tools(settings, mode="heartbeat")}
    assert "work_summary" in names
    assert not names & {"start_work", "stop_work", "log_work", "remove_work_entry"}


def test_tool_round_trip(settings, monkeypatch) -> None:
    tools = tool_map(settings)
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 9, 0))
    assert "Clock started on wakiru" in tools["start_work"].run(_ctx(settings), project="wakiru")
    _freeze(monkeypatch, settings, datetime(2026, 7, 20, 10, 0))
    out = tools["start_work"].run(_ctx(settings), project="client X")
    assert "Stopped wakiru first — 1h" in out
    assert "Stopped: client X" in tools["stop_work"].run(_ctx(settings))
    assert "Logged 2h on budget" in tools["log_work"].run(
        _ctx(settings), project="budget", minutes="120"
    )
    summary_text = tools["work_summary"].run(_ctx(settings))
    assert "budget" in summary_text
    entry = store.list_entries(settings, "budget")[0]
    assert "Removed the 2h budget entry" in tools["remove_work_entry"].run(
        _ctx(settings), id=entry.id
    )
    assert "No work timer is running" in tools["stop_work"].run(_ctx(settings))
