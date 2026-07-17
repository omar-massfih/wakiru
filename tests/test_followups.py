"""Followup-store tests — add/cancel/list, claim-exactly-once, and the tools."""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant import followups
from assistant.calendar.context import now
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_heartbeat=True,
    )


def _due_in(settings: Settings, **delta):
    return now(settings) + timedelta(**delta)


def test_add_and_list_open_sorted_by_due(settings) -> None:
    followups.add(settings, _due_in(settings, days=2), "later thing")
    followups.add(settings, _due_in(settings, hours=1), "sooner thing", "some context")
    items = followups.list_open(settings)
    assert [f.topic for f in items] == ["sooner thing", "later thing"]
    assert items[0].context == "some context"
    assert all(f.status == "open" for f in items)


def test_cancel_by_id_and_by_unambiguous_topic(settings) -> None:
    kept = followups.add(settings, _due_in(settings, hours=1), "ask about interview")
    by_topic = followups.add(settings, _due_in(settings, hours=2), "check the delivery")

    assert followups.cancel(settings, by_topic.id).id == by_topic.id
    assert followups.cancel(settings, "interview").id == kept.id
    assert followups.list_open(settings) == []


def test_cancel_ambiguous_topic_refuses(settings) -> None:
    followups.add(settings, _due_in(settings, hours=1), "check flight to Oslo")
    followups.add(settings, _due_in(settings, hours=2), "check flight to Bergen")
    assert followups.cancel(settings, "check flight") is None
    assert len(followups.list_open(settings)) == 2  # nothing guessed at


def test_cancel_unknown_returns_none(settings) -> None:
    assert followups.cancel(settings, "nope") is None
    assert followups.cancel(settings, "") is None


def test_update_by_id_changes_fields(settings) -> None:
    f = followups.add(settings, _due_in(settings, hours=1), "ask about interview", "at NAV")
    new_due = _due_in(settings, days=1)
    revised = followups.update(
        settings, f.id, due=new_due, context="waiting on their reply"
    )
    assert revised is not None
    assert revised.due == new_due.isoformat(timespec="seconds")
    assert revised.context == "waiting on their reply"
    assert revised.topic == "ask about interview"  # untouched field kept
    stored = followups.list_open(settings)[0]
    assert stored.context == "waiting on their reply" and stored.due == revised.due


def test_update_by_unambiguous_topic(settings) -> None:
    followups.add(settings, _due_in(settings, hours=1), "ask about interview")
    revised = followups.update(settings, "interview", topic="ask how the interview went")
    assert revised is not None and revised.topic == "ask how the interview went"


def test_update_ambiguous_topic_refuses(settings) -> None:
    followups.add(settings, _due_in(settings, hours=1), "check flight to Oslo")
    followups.add(settings, _due_in(settings, hours=2), "check flight to Bergen")
    assert followups.update(settings, "check flight", context="x") is None
    assert all(f.context == "" for f in followups.list_open(settings))


def test_update_with_no_fields_is_a_noop(settings) -> None:
    f = followups.add(settings, _due_in(settings, hours=1), "unchanged")
    assert followups.update(settings, f.id) is None


def test_update_cannot_touch_a_fired_or_cancelled_followup(settings) -> None:
    fired = followups.add(settings, _due_in(settings, minutes=-5), "overdue")
    followups.claim_due(settings)  # → fired
    cancelled = followups.add(settings, _due_in(settings, hours=1), "changed mind")
    followups.cancel(settings, cancelled.id)
    assert followups.update(settings, fired.id, context="x") is None
    assert followups.update(settings, cancelled.id, context="x") is None


def test_claim_due_takes_only_due_and_exactly_once(settings) -> None:
    due = followups.add(settings, _due_in(settings, minutes=-5), "overdue check-in")
    followups.add(settings, _due_in(settings, days=1), "tomorrow's check-in")

    claimed = followups.claim_due(settings)
    assert [f.id for f in claimed] == [due.id]
    assert followups.claim_due(settings) == []  # consumed
    assert [f.topic for f in followups.list_open(settings)] == ["tomorrow's check-in"]


def test_claimed_followup_is_marked_fired_not_deleted(settings) -> None:
    followups.add(settings, _due_in(settings, minutes=-5), "overdue")
    followups.claim_due(settings)
    with followups._connect(settings) as conn:
        row = conn.execute("SELECT status, fired_at FROM followups").fetchone()
    assert row["status"] == "fired" and row["fired_at"]


def test_cancelled_followup_is_never_claimed(settings) -> None:
    added = followups.add(settings, _due_in(settings, minutes=-5), "changed my mind")
    followups.cancel(settings, added.id)
    assert followups.claim_due(settings) == []


# --- the tools ---------------------------------------------------------------- #


def _run(settings: Settings, name: str, args: dict) -> str:
    from assistant.tools import ToolContext, execute_tool, tool_map

    spec = tool_map(settings)[name]
    return execute_tool(spec, ToolContext(settings=settings, thread_id="telegram:7"), args)


def test_schedule_followup_tool_roundtrip(settings) -> None:
    when = _due_in(settings, hours=3).isoformat(timespec="seconds")
    result = _run(
        settings, "schedule_followup",
        {"when": when, "topic": "ask about the viewing", "context": "Apt on Storgata"},
    )
    assert result.startswith("Follow-up scheduled: ask about the viewing")

    items = followups.list_open(settings)
    assert len(items) == 1 and items[0].thread_id == "telegram:7"

    assert "ask about the viewing" in _run(settings, "list_followups", {})
    assert _run(settings, "cancel_followup", {"target": "viewing"}).startswith("Cancelled")
    assert _run(settings, "list_followups", {}) == "No follow-ups scheduled."


def test_update_followup_tool_roundtrip(settings) -> None:
    f = followups.add(settings, _due_in(settings, hours=1), "ask about the viewing")
    result = _run(
        settings, "update_followup",
        {"target": f.id, "context": "step 2 of 3: viewing booked"},
    )
    assert result.startswith("Follow-up updated: ask about the viewing")
    assert followups.list_open(settings)[0].context == "step 2 of 3: viewing booked"


def test_update_followup_tool_refuses_no_fields_and_past_when(settings) -> None:
    f = followups.add(settings, _due_in(settings, hours=1), "carry me")
    assert "at least one of" in _run(settings, "update_followup", {"target": f.id})
    past = _due_in(settings, hours=-1).isoformat(timespec="seconds")
    assert "already in the past" in _run(
        settings, "update_followup", {"target": f.id, "when": past}
    )
    # An unresolved target reports no-match, not success.
    assert "No matching item" in _run(
        settings, "update_followup", {"target": "ghost", "context": "x"}
    )


def test_schedule_followup_rejects_bad_or_past_when(settings) -> None:
    assert "Tool failed" in _run(
        settings, "schedule_followup", {"when": "not-a-date", "topic": "x"}
    )
    past = _due_in(settings, hours=-1).isoformat(timespec="seconds")
    assert "already in the past" in _run(
        settings, "schedule_followup", {"when": past, "topic": "x"}
    )
    assert followups.list_open(settings) == []


def test_followup_tools_gated_by_heartbeat_flag(tmp_path) -> None:
    from assistant.tools import available_tools

    off = Settings(memory_dir=str(tmp_path / "m"))  # heartbeat off by default
    assert "schedule_followup" not in {t.name for t in available_tools(off)}


def test_heartbeat_mode_never_offers_send_email(tmp_path) -> None:
    from assistant.tools import available_tools

    settings = Settings(
        memory_dir=str(tmp_path / "m"),
        enable_heartbeat=True,
        enable_email=True,
        enable_email_send=True,  # even with sending explicitly on
    )
    chat_names = {t.name for t in available_tools(settings)}
    heartbeat_names = {t.name for t in available_tools(settings, mode="heartbeat")}
    assert {"send_email", "send_reply"} <= chat_names
    assert not {"send_email", "send_reply"} & heartbeat_names
    # `undo` is also chat-only: a background wake has no conversation whose
    # latest write it could revert.
    assert "undo" in chat_names and "undo" not in heartbeat_names
    # Heartbeat mode drops the send tools and undo, drops the mutating mail
    # tools while triage is not opted in (email_triage_max_actions = 0), drops
    # the chat-only doc actions (ingest/summarize spend tokens and grow docs.db),
    # and adds its own set_next_wake.
    assert "set_next_wake" in heartbeat_names and "set_next_wake" not in chat_names
    triage_only = {"reply_email", "archive_email", "mark_email_read", "label_email"}
    chat_only_docs = {"ingest_attachment", "summarize_document"}
    assert heartbeat_names == (
        chat_names - {"send_email", "send_reply", "undo"} - triage_only - chat_only_docs
    ) | {"set_next_wake"}
