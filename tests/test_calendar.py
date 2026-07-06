"""Calendar tests — store CRUD, context rendering, and the LLM write path.

The store and context run for real (plain SQLite + stdlib datetime); only the
Codex call in the write path is faked, so these stay fast and offline.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant.calendar import context, ops, store
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_auto_schedule=True,
    )


def _iso_in(settings: Settings, **delta) -> str:
    """A tz-aware ISO-8601 datetime `delta` from the assistant's current time."""
    return (context.now(settings) + timedelta(**delta)).isoformat(timespec="minutes")


# --- store ---------------------------------------------------------------- #


def test_event_roundtrip(settings) -> None:
    start = _iso_in(settings, days=1)
    event = store.create_event(
        settings, title="Dentist", start=start, location="Clinic"
    )
    assert event.id
    back = store.get_event(settings, event.id)
    assert back is not None
    assert back.title == "Dentist"
    assert back.start == start
    assert back.location == "Clinic"
    assert back.created and back.updated


def test_list_events_orders_and_bounds(settings) -> None:
    store.create_event(settings, title="Later", start=_iso_in(settings, days=3))
    store.create_event(settings, title="Sooner", start=_iso_in(settings, days=1))
    store.create_event(settings, title="WayOut", start=_iso_in(settings, days=100))

    ordered = store.list_events(settings)
    assert [e.title for e in ordered] == ["Sooner", "Later", "WayOut"]

    horizon = context.now(settings) + timedelta(days=14)
    within = store.list_events(
        settings, start_from=context.now(settings), start_to=horizon
    )
    assert [e.title for e in within] == ["Sooner", "Later"]  # WayOut excluded


def test_update_and_delete(settings) -> None:
    event = store.create_event(settings, title="Lunch", start=_iso_in(settings, days=1))
    new_start = _iso_in(settings, days=1, hours=1)
    revised = store.update_event(settings, event.id, start=new_start, location="Cafe")
    assert revised is not None and revised.start == new_start and revised.location == "Cafe"

    assert store.delete_event(settings, event.id) is not None
    assert store.get_event(settings, event.id) is None
    assert store.delete_event(settings, event.id) is None  # already gone


def test_find_event_by_id_and_title(settings) -> None:
    event = store.create_event(
        settings, title="Dentist appointment", start=_iso_in(settings, days=2)
    )
    assert store.find_event(settings, event.id).id == event.id  # by id
    assert store.find_event(settings, "dentist").id == event.id  # fuzzy title
    assert store.find_event(settings, "nonexistent") is None


def test_find_event_prefers_soonest_upcoming(settings) -> None:
    store.create_event(settings, title="Standup", start=_iso_in(settings, days=5))
    soon = store.create_event(settings, title="Standup", start=_iso_in(settings, days=1))
    assert store.find_event(settings, "standup").id == soon.id


# --- context (read path) -------------------------------------------------- #


def test_agenda_context_has_time_and_events(settings) -> None:
    store.create_event(settings, title="Team sync", start=_iso_in(settings, days=1))
    block = context.agenda_context(settings)
    assert "Current date and time" in block
    assert str(context.now(settings).year) in block
    assert "Team sync" in block


def test_agenda_context_empty_calendar(settings) -> None:
    block = context.agenda_context(settings)
    assert "no upcoming events" in block.lower()


def test_render_events_can_expose_ids(settings) -> None:
    event = store.create_event(settings, title="Review", start=_iso_in(settings, days=1))
    upcoming = context.upcoming_events(settings)
    with_ids = context.render_events(settings, upcoming, with_ids=True)
    without = context.render_events(settings, upcoming, with_ids=False)
    assert event.id in with_ids and event.id not in without


# --- ops (write path) ----------------------------------------------------- #


def test_parse_ops_filters_and_strips_fences() -> None:
    raw = (
        "```json\n"
        '[{"op": "create", "title": "X", "start": "2026-07-10T15:00:00+02:00"},'
        ' {"op": "bogus"},'
        ' {"op": "cancel", "id": "abc"}]\n'
        "```"
    )
    parsed = ops._parse_ops(raw)
    assert [op["op"] for op in parsed] == ["create", "cancel"]
    assert ops._parse_ops("not json at all") == []


def test_update_calendar_create_reschedule_cancel(settings, monkeypatch) -> None:
    start = _iso_in(settings, days=2)
    canned_create = f'[{{"op": "create", "title": "Dentist", "start": "{start}"}}]'
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned_create)

    applied = ops.update_calendar(settings, "book the dentist friday", "Done!")
    assert any(s.startswith("created:") for s in applied)
    events = store.list_events(settings)
    assert len(events) == 1
    event_id = events[0].id

    new_start = _iso_in(settings, days=2, hours=1)
    canned_move = f'[{{"op": "reschedule", "id": "{event_id}", "start": "{new_start}"}}]'
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned_move)
    ops.update_calendar(settings, "move it an hour later", "Moved.")
    assert store.get_event(settings, event_id).start == new_start

    canned_cancel = f'[{{"op": "cancel", "id": "{event_id}"}}]'
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned_cancel)
    ops.update_calendar(settings, "cancel the dentist", "Cancelled.")
    assert store.list_events(settings) == []


def test_update_calendar_disabled_is_noop(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"), enable_auto_schedule=False
    )
    monkeypatch.setattr(
        "assistant.calendar.ops.run_codex",
        lambda *a, **k: pytest.fail("run_codex must not be called when disabled"),
    )
    assert ops.update_calendar(settings, "book something", "ok") == []
