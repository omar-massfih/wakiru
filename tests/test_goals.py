"""Goal-store tests — open/update/close, the cap, raising, tools, and context."""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant import goals, heartbeat
from assistant.calendar.context import now
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_heartbeat=True,
    )


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )


def _at(settings: Settings, **delta):
    return now(settings) + timedelta(**delta)


# --- the store ---------------------------------------------------------------- #


def test_open_and_list_sorted_by_next_action(settings) -> None:
    goals.open_goal(settings, "later project", "plan", _at(settings, days=2))
    goals.open_goal(settings, "parked project", "waiting")
    goals.open_goal(settings, "sooner project", "step 1", _at(settings, hours=1))
    items = goals.list_open(settings)
    assert [g.title for g in items] == [
        "sooner project", "later project", "parked project",
    ]
    assert all(g.status == "open" for g in items)


def test_open_goal_refuses_past_the_cap(settings) -> None:
    for i in range(settings.goals_max_open):
        assert goals.open_goal(settings, f"goal {i}", "s") is not None
    assert goals.open_goal(settings, "one too many", "s") is None
    assert len(goals.list_open(settings)) == settings.goals_max_open


def test_state_is_clipped_to_the_cap(settings) -> None:
    saved = goals.open_goal(settings, "big plans", "x" * 5000)
    assert saved is not None and len(saved.state) == goals.STATE_MAX_CHARS
    revised = goals.update(settings, saved.id, state="y" * 5000)
    assert revised is not None and len(revised.state) == goals.STATE_MAX_CHARS


def test_update_rewrites_state_and_moves_next_action(settings) -> None:
    g = goals.open_goal(settings, "plan the Oslo trip", "step 1: research hotels")
    later = _at(settings, days=1)
    revised = goals.update(
        settings, g.id, state="step 2: shortlist ready", next_action_at=later
    )
    assert revised is not None
    assert revised.state == "step 2: shortlist ready"
    assert revised.next_action_at == later.isoformat(timespec="seconds")
    stored = goals.list_open(settings)[0]
    assert stored.state == revised.state and stored.updated_at >= g.updated_at


def test_update_can_park_a_goal(settings) -> None:
    g = goals.open_goal(settings, "waiting game", "s", _at(settings, hours=1))
    revised = goals.update(settings, g.id, clear_next_action=True)
    assert revised is not None and revised.next_action_at == ""
    assert goals.ready(settings, _at(settings, days=1)) == []


def test_update_by_unambiguous_title_and_refuses_ambiguity(settings) -> None:
    goals.open_goal(settings, "plan the Oslo trip", "s")
    assert goals.update(settings, "oslo", state="new") is not None
    goals.open_goal(settings, "plan the Bergen trip", "s")
    assert goals.update(settings, "plan the", state="x") is None


def test_update_with_no_fields_is_a_noop(settings) -> None:
    g = goals.open_goal(settings, "unchanged", "s")
    assert goals.update(settings, g.id) is None


def test_close_marks_done_and_records_outcome(settings) -> None:
    g = goals.open_goal(settings, "find a better power deal", "comparing")
    closed = goals.close(settings, g.id, "switched to Tibber")
    assert closed is not None and closed.status == "done"
    assert goals.list_open(settings) == []
    with goals._connect(settings) as conn:
        row = conn.execute("SELECT status, outcome FROM goals").fetchone()
    assert row["status"] == "done" and row["outcome"] == "switched to Tibber"


def test_close_abandoned_and_cannot_touch_a_closed_goal(settings) -> None:
    g = goals.open_goal(settings, "dead end", "s")
    assert goals.close(settings, g.id, "not worth it", abandoned=True).status == "abandoned"
    assert goals.close(settings, g.id) is None
    assert goals.update(settings, g.id, state="zombie") is None


def test_ready_returns_only_due_goals_without_consuming_them(settings) -> None:
    due = goals.open_goal(settings, "due goal", "s", _at(settings, minutes=-5))
    goals.open_goal(settings, "future goal", "s", _at(settings, days=1))
    goals.open_goal(settings, "parked goal", "s")
    assert [g.id for g in goals.ready(settings)] == [due.id]
    # Standing, not claimed: still ready on the next look.
    assert [g.id for g in goals.ready(settings)] == [due.id]


def test_stale_flags_only_old_untouched_goals(settings) -> None:
    g = goals.open_goal(settings, "old project", "s")
    assert goals.stale(settings) == []
    later = now(settings) + timedelta(days=settings.goal_stale_days + 1)
    assert [x.id for x in goals.stale(settings, later)] == [g.id]


# --- heartbeat integration ---------------------------------------------------- #


def test_ready_goal_is_raised_and_counts_as_scheduled(settings) -> None:
    goals.open_goal(settings, "plan the Oslo trip", "book hotel", _at(settings, minutes=-5))
    situation = heartbeat.gather_situation(settings)
    assert situation is not None and situation.scheduled
    assert [g.title for g in situation.goals] == ["plan the Oslo trip"]
    assert "Goal ready for its next step: plan the Oslo trip" in situation.report()


def test_ignored_goal_reraises_only_after_the_cadence_window(settings) -> None:
    goals.open_goal(settings, "stuck goal", "s", _at(settings, minutes=-5))
    current = now(settings)
    assert len(heartbeat._raisable_goals(settings, current)) == 1
    # A short self-paced wake soon after: the untouched goal stays quiet.
    assert heartbeat._raisable_goals(settings, current + timedelta(minutes=5)) == []
    # After a full base-cadence window it comes back.
    later = current + timedelta(minutes=settings.heartbeat_minutes)
    assert len(heartbeat._raisable_goals(settings, later)) == 1


def test_advanced_goal_raises_immediately_again(settings, monkeypatch) -> None:
    g = goals.open_goal(settings, "moving goal", "s", _at(settings, minutes=-5))
    current = now(settings)
    assert len(heartbeat._raisable_goals(settings, current)) == 1
    # The model advances the goal a little later in the wake: a changed
    # updated_at re-raises it right away, no cadence wait.
    monkeypatch.setattr("assistant.goals.now", lambda s: current + timedelta(seconds=30))
    goals.update(settings, g.id, state="advanced", next_action_at=current)
    assert len(heartbeat._raisable_goals(settings, current + timedelta(minutes=1))) == 1


def test_goal_next_action_pulls_the_next_wake(settings) -> None:
    current = now(settings)
    soon = (current + timedelta(minutes=20)).replace(microsecond=0)
    goals.open_goal(settings, "timed step", "s", soon)
    assert heartbeat.next_wake_at(settings, current) == soon


def test_stale_goal_nudge_lands_in_the_report(settings) -> None:
    goals.open_goal(settings, "dusty project", "s")
    with goals._connect(settings) as conn:
        old = (now(settings) - timedelta(days=settings.goal_stale_days + 1)).isoformat(
            timespec="seconds"
        )
        conn.execute("UPDATE goals SET updated_at = ?, created_at = ?", (old, old))
    situation = heartbeat.gather_situation(settings)
    assert situation is not None and not situation.scheduled
    assert "Stale goal: dusty project" in situation.report()


# --- the tools ---------------------------------------------------------------- #


def _run(settings: Settings, name: str, args: dict) -> str:
    from assistant.tools import ToolContext, execute_tool, tool_map

    spec = tool_map(settings)[name]
    return execute_tool(spec, ToolContext(settings=settings, thread_id="telegram:7"), args)


def test_goal_tool_roundtrip(settings) -> None:
    when = _at(settings, hours=3).isoformat(timespec="seconds")
    result = _run(
        settings, "open_goal",
        {"title": "Plan the Oslo trip", "state": "step 1: dates", "next_action": when},
    )
    assert result.startswith("Goal opened: Plan the Oslo trip")
    assert goals.list_open(settings)[0].thread_id == "telegram:7"

    listing = _run(settings, "list_goals", {})
    assert "Plan the Oslo trip" in listing and "step 1: dates" in listing

    assert _run(
        settings, "update_goal", {"target": "oslo", "state": "step 2: hotels", "park": True}
    ).startswith("Goal updated")
    stored = goals.list_open(settings)[0]
    assert stored.state == "step 2: hotels" and stored.next_action_at == ""

    assert _run(
        settings, "close_goal", {"target": "oslo", "outcome": "booked"}
    ) == "Goal done: Plan the Oslo trip"
    assert _run(settings, "list_goals", {}) == "No open goals."


def test_goal_tools_validate_input(settings) -> None:
    assert "Tool failed" in _run(
        settings, "open_goal", {"title": "x", "state": "s", "next_action": "not-a-date"}
    )
    g = goals.open_goal(settings, "carry me", "s")
    assert "at least one of" in _run(settings, "update_goal", {"target": g.id})
    assert "No matching item" in _run(
        settings, "update_goal", {"target": "ghost", "state": "x"}
    )
    for i in range(settings.goals_max_open - 1):
        goals.open_goal(settings, f"filler {i}", "s")
    assert "already carry" in _run(
        settings, "open_goal", {"title": "over cap", "state": "s"}
    )


def test_goal_tools_gated_by_heartbeat_flag(tmp_path) -> None:
    from assistant.tools import available_tools

    off = Settings(memory_dir=str(tmp_path / "m"))  # heartbeat off by default
    assert "open_goal" not in {t.name for t in available_tools(off)}
    on = Settings(memory_dir=str(tmp_path / "m2"), enable_heartbeat=True)
    names = {t.name for t in available_tools(on, mode="heartbeat")}
    assert {"open_goal", "update_goal", "close_goal", "list_goals"} <= names


# --- the context provider ----------------------------------------------------- #


def test_goals_context_block_carries_state(settings) -> None:
    from assistant.context_providers import build_context

    goals.open_goal(settings, "Plan the Oslo trip", "step 2: hotels shortlisted")
    blocks = build_context(settings, "anything", "telegram:7")
    assert "Plan the Oslo trip" in blocks["goals"]
    assert "step 2: hotels shortlisted" in blocks["goals"]


def test_goals_context_block_absent_without_heartbeat(tmp_path) -> None:
    from assistant.context_providers import build_context

    settings = Settings(memory_dir=str(tmp_path / "m"))
    assert "goals" not in build_context(settings, "anything", "t")
