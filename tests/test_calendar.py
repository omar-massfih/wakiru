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


def test_format_when(settings) -> None:
    assert context.format_when(settings, "2026-07-08T23:00:00+00:00") == "Thu 09 Jul 2026 01:00"
    assert "2026-07-08T23:00:00" not in context.format_when(settings, "2026-07-08T23:00:00+00:00")
    assert context.format_when(settings, "not-a-date") == "not-a-date"


def test_apply_op_summary_human_dates(settings) -> None:
    summary = ops.apply_op(
        settings,
        {"op": "create", "title": "Get ready for bed", "start": "2026-07-08T23:00:00+00:00"},
    )
    assert summary == "created: Get ready for bed @ Thu 09 Jul 2026 01:00"
    assert "2026-07-08T23:00:00" not in summary


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


# --- per-occurrence exceptions (skip / move) ------------------------------ #


def _weekly_series(settings: Settings) -> store.Event:
    return store.create_event(
        settings,
        title="Standup",
        start=_past_monday(settings).isoformat(timespec="minutes"),
        rrule="FREQ=WEEKLY;BYDAY=MO",
    )


def _upcoming_mondays(settings: Settings) -> list:
    current = context.now(settings)
    return recurrence.occurrences_in(settings, current, current + timedelta(days=28))


def test_resolve_occurrence_snaps_to_series(settings) -> None:
    series = _weekly_series(settings)
    real = store.parse_dt(_upcoming_mondays(settings)[0].start)
    assert recurrence.resolve_occurrence(series, real) == real  # exact
    loose = real.replace(hour=15, minute=30)  # same date, sloppy time
    assert recurrence.resolve_occurrence(series, loose) == real


def test_skip_occurrence_drops_only_that_date(settings, monkeypatch) -> None:
    series = _weekly_series(settings)
    mondays = _upcoming_mondays(settings)
    first, second = store.parse_dt(mondays[0].start), store.parse_dt(mondays[1].start)

    canned = f'[{{"op": "skip", "id": "{series.id}", "occurrence": "{first.isoformat()}"}}]'
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned)
    applied = ops.update_calendar(settings, "skip this monday's standup", "Skipped.")
    assert any(s.startswith("skipped:") for s in applied)

    starts = [store.parse_dt(e.start) for e in _upcoming_mondays(settings)]
    assert first not in starts and second in starts  # only the one date is gone


def test_move_occurrence_changes_only_that_one(settings, monkeypatch) -> None:
    series = _weekly_series(settings)
    mondays = _upcoming_mondays(settings)
    first = store.parse_dt(mondays[0].start)
    new_start = (first + timedelta(days=1, hours=1)).isoformat()  # Tuesday, an hour later

    canned = (
        f'[{{"op": "move", "id": "{series.id}", "occurrence": "{first.isoformat()}",'
        f' "start": "{new_start}", "title": "Standup (special)"}}]'
    )
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned)
    applied = ops.update_calendar(settings, "move this monday's standup to tuesday", "Moved.")
    assert any(s.startswith("moved:") for s in applied)

    occ = _upcoming_mondays(settings)
    moved = [e for e in occ if e.title == "Standup (special)"]
    assert len(moved) == 1 and moved[0].start == new_start
    assert all(store.parse_dt(e.start) != first for e in occ)  # original slot vacated
    assert all(e.title == "Standup" for e in occ if e.title != "Standup (special)")


def test_exception_helpers_reject_non_series(settings) -> None:
    one_shot = store.create_event(settings, title="Once", start=_iso_in(settings, days=1))
    assert store.add_exdate(settings, one_shot.id, one_shot.start) is None
    assert store.set_override(settings, one_shot.id, one_shot.start, {"title": "x"}) is None


# --- naive-datetime hardening ---------------------------------------------- #


def test_naive_start_is_normalized_on_write(settings) -> None:
    # The extractor is told to emit offsets, but an LLM slip must not poison the
    # store: one naive start used to TypeError every aware/naive comparison.
    naive = (context.now(settings) + timedelta(days=1)).replace(tzinfo=None)
    event = store.create_event(settings, title="Dentist", start=naive.isoformat())
    assert store.parse_dt(event.start).tzinfo is not None

    moved = store.update_event(
        settings, event.id, start=(naive + timedelta(hours=1)).isoformat()
    )
    assert store.parse_dt(moved.start).tzinfo is not None
    # The agenda read path works with the event in place.
    assert "Dentist" in context.agenda_context(settings)


def test_legacy_naive_row_does_not_break_reads(settings) -> None:
    # A pre-normalization row (or a hand-edit) lands naive in the DB; reads must
    # survive it — parse_dt treats naive as local instead of raising.
    naive = (context.now(settings) + timedelta(days=1)).replace(tzinfo=None)
    with store._connect(settings) as conn:
        conn.execute(
            "INSERT INTO events (id, title, start, rrule) VALUES (?, ?, ?, ?)",
            ("legacy1", "Old one-shot", naive.isoformat(), ""),
        )
        conn.execute(
            "INSERT INTO events (id, title, start, rrule) VALUES (?, ?, ?, ?)",
            ("legacy2", "Old series", naive.isoformat(), "FREQ=DAILY"),
        )

    horizon = context.now(settings) + timedelta(days=7)
    bounded = store.list_events(settings, start_from=context.now(settings), start_to=horizon)
    assert {e.title for e in bounded} == {"Old one-shot", "Old series"}

    occurrences = recurrence.occurrences_in(settings, context.now(settings), horizon)
    assert any(e.title == "Old series" for e in occurrences)
    assert "Old one-shot" in context.agenda_context(settings)


def test_weekly_series_keeps_wall_clock_across_dst(settings) -> None:
    # A master created in winter round-trips through ISO with a fixed +01:00
    # offset; summer occurrences must still land at 09:00 Oslo wall time
    # (+02:00), not stay pinned to the winter offset (which would render 10:00).
    from datetime import datetime
    from zoneinfo import ZoneInfo

    oslo = ZoneInfo("Europe/Oslo")
    master = store.create_event(
        settings,
        title="Standup",
        start="2026-01-05T09:00:00+01:00",  # a Monday, CET
        rrule="FREQ=WEEKLY;BYDAY=MO",
    )
    occurrences = recurrence.expand(
        master,
        datetime(2026, 7, 6, 0, 0, tzinfo=oslo),
        datetime(2026, 7, 12, 23, 59, tzinfo=oslo),
        oslo,
    )
    assert len(occurrences) == 1
    occ = store.parse_dt(occurrences[0].start).astimezone(oslo)
    assert (occ.hour, occ.minute) == (9, 0)
    assert occ.utcoffset() == timedelta(hours=2)  # CEST — same wall clock, new offset
