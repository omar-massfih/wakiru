"""Calendar tests — store CRUD, context rendering, and the LLM write path.

The store and context run for real (plain SQLite + stdlib datetime); only the
Codex call in the write path is faked, so these stay fast and offline.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from assistant.calendar import context, ops, recurrence, store
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


def _past_monday(settings: Settings, weeks_ago: int = 3, hour: int = 9):
    """A Monday `weeks_ago` weeks back at `hour`:00 — a DTSTART already in the past."""
    current = context.now(settings)
    monday = current - timedelta(days=current.weekday(), weeks=weeks_ago)
    return monday.replace(hour=hour, minute=0, second=0, microsecond=0)


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


def test_update_calendar_creates_recurring_series(settings, monkeypatch) -> None:
    start = _iso_in(settings, days=1)
    canned = (
        f'[{{"op": "create", "title": "Standup", "start": "{start}",'
        ' "rrule": "FREQ=WEEKLY;BYDAY=MO"}]'
    )
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned)

    applied = ops.update_calendar(settings, "standup every monday", "Set up.")
    assert any("every Monday" in s for s in applied)
    events = store.list_events(settings)
    assert len(events) == 1 and events[0].rrule == "FREQ=WEEKLY;BYDAY=MO"


def test_create_drops_invalid_rrule_but_keeps_event(settings, monkeypatch) -> None:
    start = _iso_in(settings, days=1)
    canned = f'[{{"op": "create", "title": "Thing", "start": "{start}", "rrule": "FREQ=NEVER"}}]'
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned)

    ops.update_calendar(settings, "schedule thing", "Done.")
    events = store.list_events(settings)
    assert len(events) == 1 and events[0].rrule == ""  # rule dropped, event kept


# --- recurrence (expansion & migration) ----------------------------------- #


def test_event_persists_rrule(settings) -> None:
    event = store.create_event(
        settings, title="1:1", start=_iso_in(settings, days=1), rrule="FREQ=WEEKLY;BYDAY=MO"
    )
    assert store.get_event(settings, event.id).rrule == "FREQ=WEEKLY;BYDAY=MO"


def test_migration_adds_rrule_to_legacy_db(settings) -> None:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.calendar_db_path) as conn:  # pre-rrule schema
        conn.execute(
            "CREATE TABLE events (id TEXT PRIMARY KEY, title TEXT NOT NULL,"
            " start TEXT NOT NULL, end TEXT DEFAULT '', location TEXT DEFAULT '',"
            " notes TEXT DEFAULT '', created TEXT DEFAULT '', updated TEXT DEFAULT '')"
        )
        conn.execute(
            "INSERT INTO events (id, title, start) VALUES"
            " ('legacy1', 'Old', '2020-01-01T09:00:00+01:00')"
        )

    back = store.get_event(settings, "legacy1")  # _connect adds the column
    assert back is not None and back.rrule == ""
    new = store.create_event(
        settings, title="New", start="2030-01-01T09:00:00+01:00", rrule="FREQ=DAILY"
    )
    assert store.get_event(settings, new.id).rrule == "FREQ=DAILY"


def test_occurrences_expand_weekly_past_dtstart(settings) -> None:
    start_dt = _past_monday(settings)
    store.create_event(
        settings,
        title="Standup",
        start=start_dt.isoformat(timespec="minutes"),
        end=(start_dt + timedelta(minutes=30)).isoformat(timespec="minutes"),
        rrule="FREQ=WEEKLY;BYDAY=MO",
    )

    current = context.now(settings)
    occ = recurrence.occurrences_in(settings, current, current + timedelta(days=14))
    assert occ  # a past-DTSTART series still yields upcoming occurrences
    for e in occ:
        dt = store.parse_dt(e.start)
        assert dt.weekday() == 0 and dt >= current  # upcoming Mondays only
        assert e.rrule == ""  # an occurrence, not the series master
        assert store.parse_dt(e.end) - dt == timedelta(minutes=30)  # duration kept


def test_upcoming_events_expands_series(settings) -> None:
    store.create_event(
        settings,
        title="Weekly",
        start=_past_monday(settings).isoformat(timespec="minutes"),
        rrule="FREQ=WEEKLY;BYDAY=MO",
    )
    assert any(e.title == "Weekly" for e in context.upcoming_events(settings))


def test_writer_view_shows_series_with_summary(settings) -> None:
    event = store.create_event(
        settings,
        title="1:1",
        start=_past_monday(settings).isoformat(timespec="minutes"),
        rrule="FREQ=WEEKLY;BYDAY=MO",
    )
    view = context.writer_view(settings)
    assert any(e.id == event.id for e in view)  # series shown despite past DTSTART
    rendered = context.render_events(settings, view, with_ids=True)
    assert "every Monday" in rendered and event.id in rendered


def test_humanize_rrule() -> None:
    assert recurrence.humanize_rrule("FREQ=WEEKLY;BYDAY=MO") == "every Monday"
    assert recurrence.humanize_rrule("FREQ=DAILY") == "daily"
    assert recurrence.humanize_rrule("FREQ=WEEKLY;INTERVAL=2") == "every 2 weeks"
