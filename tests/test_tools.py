"""Tool registry tests — gating, dispatch, and the shared guarded write paths.

Calendar/task tools run against real tmp-path SQLite stores (so the undo ledger
integration is exercised for real); memory tools use the same fake bag-of-words
embedder as test_memory.py. No LLM anywhere: tools are deterministic code.
"""

from __future__ import annotations

import math
import re
import zlib
from datetime import timedelta

import pytest

from assistant.calendar import context
from assistant.calendar import store as calendar_store
from assistant.config import Settings
from assistant.tasks import store as tasks_store
from assistant.tools import ToolContext, available_tools, execute_tool, tool_map
from assistant.undo import undo_latest

THREAD = "telegram:1"


def _fake_embed(texts: list[str], prefix: str = "", settings=None) -> list[list[float]]:
    """The same bag-of-words fake test_memory.py uses (word overlap -> high cosine)."""
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_write_confirmation=True,
        dedup_threshold=0.8,
        forget_threshold=0.4,
        forget_ambiguity_margin=0.1,
    )


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    monkeypatch.setattr("assistant.memory.embeddings._embed", _fake_embed)


def _ctx(settings: Settings) -> ToolContext:
    return ToolContext(settings=settings, thread_id=THREAD, batch_id="batch-1")


def _iso_in(settings: Settings, **delta) -> str:
    return (context.now(settings) + timedelta(**delta)).isoformat(timespec="minutes")


# --- registry gating -------------------------------------------------------- #


def test_default_registry_has_no_email_tools() -> None:
    names = {spec.name for spec in available_tools(Settings())}
    assert "add_task" in names and "create_event" in names and "remember" in names
    assert not any(name.endswith("email") for name in names)


def test_email_tools_appear_without_send_unless_gated() -> None:
    names = {s.name for s in available_tools(Settings(enable_email=True))}
    assert {"list_email", "read_email", "draft_email"} <= names
    assert "send_email" not in names  # second switch off

    gated = {
        s.name
        for s in available_tools(Settings(enable_email=True, enable_email_send=True))
    }
    assert "send_email" in gated


def test_disabled_subsystems_drop_their_tools() -> None:
    names = {
        s.name
        for s in available_tools(
            Settings(enable_calendar=False, enable_tasks=False, enable_docs=False)
        )
    }
    assert "create_event" not in names
    assert "add_task" not in names
    assert "search_documents" not in names
    assert "remember" in names  # memory tools are unconditional


# --- dispatch --------------------------------------------------------------- #


def test_execute_tool_missing_required_arg_is_error_string(settings) -> None:
    spec = tool_map(settings)["add_task"]
    result = execute_tool(spec, _ctx(settings), {})
    assert result.startswith("Tool failed: missing required")
    assert tasks_store.list_tasks(settings) == []


def test_execute_tool_exception_becomes_result_string(settings, monkeypatch) -> None:
    spec = tool_map(settings)["add_task"]
    monkeypatch.setattr(
        "assistant.tasks.ops.apply_op",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")),
    )
    result = execute_tool(spec, _ctx(settings), {"title": "x"})
    assert result.startswith("Tool failed:")


def test_unknown_args_are_dropped_not_fatal(settings) -> None:
    spec = tool_map(settings)["add_task"]
    result = execute_tool(spec, _ctx(settings), {"title": "Buy milk", "bogus": 1})
    assert result == "added task: Buy milk"


# --- calendar/tasks write paths (undo ledger included) ----------------------- #


def test_create_event_writes_store_and_undo_ledger(settings) -> None:
    spec = tool_map(settings)["create_event"]
    result = execute_tool(
        spec, _ctx(settings), {"title": "Dentist", "start": _iso_in(settings, days=2)}
    )
    assert result.startswith("created: Dentist")
    assert [e.title for e in calendar_store.list_events(settings)] == ["Dentist"]

    undone = undo_latest(settings, THREAD, settings.write_undo_window_minutes)
    assert undone.startswith("Undone:")
    assert calendar_store.list_events(settings) == []


def test_task_roundtrip_and_ambiguous_target(settings) -> None:
    ctx = _ctx(settings)
    tools = tool_map(settings)
    assert execute_tool(tools["add_task"], ctx, {"title": "Buy milk"}).startswith("added")
    task = tasks_store.list_tasks(settings)[0]

    missing = execute_tool(tools["complete_task"], ctx, {"id": "no-such-task"})
    assert "No matching item" in missing

    done = execute_tool(tools["complete_task"], ctx, {"id": task.id})
    assert done == f"completed: {task.title}"
    assert tasks_store.list_tasks(settings) == []  # open tasks only


# --- reminder mutes ----------------------------------------------------------- #


def test_reminder_tools_gated_on_enable_reminders() -> None:
    on = {s.name for s in available_tools(Settings())}
    assert {"mute_reminders", "unmute_reminders"} <= on
    off = {s.name for s in available_tools(Settings(enable_reminders=False))}
    assert not {"mute_reminders", "unmute_reminders"} & off


def test_mute_by_title_defaults_to_end_of_today(settings) -> None:
    from assistant import mutes

    event = calendar_store.create_event(
        settings, title="Exercise", start=_iso_in(settings, minutes=30)
    )
    result = execute_tool(
        tool_map(settings)["mute_reminders"], _ctx(settings), {"target": "Exercise"}
    )
    assert result.startswith("Muted reminders for Exercise until")

    current = context.now(settings)
    active = mutes.active_mutes(settings, current)
    assert set(active) == {("event", event.id)}
    until = active[("event", event.id)]
    assert until.date() == current.date() and (until.hour, until.minute) == (23, 59)


def test_mute_all_and_unmute(settings) -> None:
    from assistant import mutes

    ctx = _ctx(settings)
    tools = tool_map(settings)
    assert execute_tool(tools["mute_reminders"], ctx, {"target": "all"}).startswith(
        "Muted reminders for all reminders"
    )
    assert mutes.all_muted(settings, context.now(settings))

    assert execute_tool(tools["unmute_reminders"], ctx, {"target": "all"}) == (
        "Unmuted reminders for all reminders."
    )
    assert not mutes.all_muted(settings, context.now(settings))
    assert execute_tool(tools["unmute_reminders"], ctx, {"target": "all"}) == (
        "No active mute for all reminders."
    )


def test_mute_resolves_tasks_and_refuses_ambiguity(settings) -> None:
    from assistant import mutes

    ctx = _ctx(settings)
    tools = tool_map(settings)
    task = tasks_store.create_task(settings, "Pay bill", due=_iso_in(settings, hours=2))
    assert execute_tool(tools["mute_reminders"], ctx, {"target": "Pay bill"}).startswith(
        "Muted reminders for Pay bill"
    )
    assert set(mutes.active_mutes(settings, context.now(settings))) == {("task", task.id)}

    calendar_store.create_event(settings, title="Gym A", start=_iso_in(settings, hours=1))
    calendar_store.create_event(settings, title="Gym B", start=_iso_in(settings, hours=2))
    assert "No matching item" in execute_tool(
        tools["mute_reminders"], ctx, {"target": "Gym"}
    )
    assert "No matching item" in execute_tool(
        tools["mute_reminders"], ctx, {"target": "no such thing"}
    )


def test_mute_rejects_bad_until(settings) -> None:
    calendar_store.create_event(settings, title="Exercise", start=_iso_in(settings, hours=1))
    ctx = _ctx(settings)
    tools = tool_map(settings)
    assert execute_tool(
        tools["mute_reminders"], ctx, {"target": "Exercise", "until": "not-a-date"}
    ).startswith("Tool failed: until must be")
    assert execute_tool(
        tools["mute_reminders"],
        ctx,
        {"target": "Exercise", "until": _iso_in(settings, hours=-1)},
    ) == "Tool failed: until is already in the past."


# --- memory tools ------------------------------------------------------------ #


def test_remember_search_forget_roundtrip(settings) -> None:
    ctx = _ctx(settings)
    tools = tool_map(settings)

    saved = execute_tool(
        tools["remember"],
        ctx,
        {"content": "The user prefers Norwegian replies.", "profile": True},
    )
    assert saved.startswith("Saved:")

    found = execute_tool(tools["search_memory"], ctx, {"query": "Norwegian replies"})
    assert "Norwegian" in found

    forgot = execute_tool(
        tools["forget"], ctx, {"target": "The user prefers Norwegian replies."}
    )
    assert forgot.startswith("Forgot:")

    gone = execute_tool(tools["forget"], ctx, {"target": "completely unrelated thing"})
    assert gone.startswith("No memory matched")


def test_remember_profile_tag_reaches_profile_context(settings) -> None:
    from assistant.memory.profile import profile_context

    execute_tool(
        tool_map(settings)["remember"],
        _ctx(settings),
        {"content": "The user works 09:00-17:00 in Bergen.", "profile": True},
    )
    assert "Bergen" in profile_context(settings)


# --- the undo tool ------------------------------------------------------------ #


def test_undo_tool_gated_on_write_confirmation_and_a_writable_subsystem() -> None:
    assert "undo" in {s.name for s in available_tools(Settings())}
    assert "undo" not in {
        s.name for s in available_tools(Settings(enable_write_confirmation=False))
    }
    assert "undo" not in {
        s.name
        for s in available_tools(
            Settings(enable_calendar=False, enable_tasks=False)
        )
    }
    # And never in the background: a heartbeat wake has no conversation.
    assert "undo" not in {
        s.name for s in available_tools(Settings(), mode="heartbeat")
    }


def test_undo_tool_reverts_the_latest_write_on_the_thread(settings) -> None:
    execute_tool(
        tool_map(settings)["create_event"],
        _ctx(settings),
        {"title": "Dentist", "start": _iso_in(settings, days=2)},
    )
    result = execute_tool(tool_map(settings)["undo"], _ctx(settings), {})
    assert result.startswith("Undone:")
    assert calendar_store.list_events(settings) == []


def test_undo_tool_with_nothing_to_revert_says_so(settings) -> None:
    result = execute_tool(tool_map(settings)["undo"], _ctx(settings), {})
    assert not result.startswith("Undone:")
    assert result  # a user-ready explanation, never empty
