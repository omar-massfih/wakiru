"""Undo ledger tests — restore_event, record_write, and undo_latest.

Everything runs for real (plain SQLite + stdlib datetime); writes are exercised
by applying parsed operations directly through ``ops.apply_op`` — exactly what
the calendar tools do — matching test_calendar.py.
"""

from __future__ import annotations

import uuid
from datetime import UTC, timedelta

import pytest

from assistant.calendar import context, ops, store, undo
from assistant.config import Settings

THREAD = "telegram:1"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_write_confirmation=True,
        write_undo_window_minutes=15,
    )


def _iso_in(settings: Settings, **delta) -> str:
    return (context.now(settings) + timedelta(**delta)).isoformat(timespec="minutes")


def _apply(settings: Settings, ops_list: list[dict], thread: str = THREAD) -> list[str]:
    """Apply parsed operations as the calendar tools do — one undo batch per call."""
    batch_id = uuid.uuid4().hex
    applied = []
    for op in ops_list:
        result = ops.apply_op(settings, op, thread, batch_id)
        if result:
            applied.append(result)
    return applied


def _past_monday(settings: Settings, weeks_ago: int = 3, hour: int = 9):
    current = context.now(settings)
    monday = current - timedelta(days=current.weekday(), weeks=weeks_ago)
    return monday.replace(hour=hour, minute=0, second=0, microsecond=0)


def _weekly_series(settings: Settings) -> store.Event:
    return store.create_event(
        settings,
        title="Standup",
        start=_past_monday(settings).isoformat(timespec="minutes"),
        rrule="FREQ=WEEKLY;BYDAY=MO",
    )


# --- store.restore_event --------------------------------------------------- #


def test_restore_event_recreates_deleted_event(settings) -> None:
    event = store.create_event(settings, title="Dentist", start=_iso_in(settings, days=1))
    deleted = store.delete_event(settings, event.id)
    assert store.get_event(settings, event.id) is None

    restored = store.restore_event(settings, deleted)
    back = store.get_event(settings, event.id)
    assert back is not None
    assert back == deleted == restored


def test_restore_event_does_not_bump_updated(settings) -> None:
    event = store.create_event(settings, title="Lunch", start=_iso_in(settings, days=1))
    original_updated = event.updated
    store.update_event(settings, event.id, start=_iso_in(settings, days=1, hours=1))

    store.restore_event(settings, event)
    back = store.get_event(settings, event.id)
    assert back.updated == original_updated  # restored verbatim, not re-stamped
    assert back.start == event.start


# --- undo.record_write ------------------------------------------------------ #


def _log_rows(settings: Settings) -> list[dict]:
    with undo._connect(settings) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM write_log").fetchall()]


def test_record_write_noop_without_thread_id(settings) -> None:
    undo.record_write(settings, "", "batch", "evt", "create", "created: x", None)
    assert _log_rows(settings) == []


# --- undo.undo_latest via the ops write path -------------------------------- #


def test_undo_latest_nothing_to_undo_when_ledger_empty(settings) -> None:
    assert undo.undo_latest(settings, THREAD, 15) == "Nothing to undo."


def test_undo_latest_reverts_create(settings) -> None:
    start = _iso_in(settings, days=2)
    _apply(settings, [{"op": "create", "title": "Dentist", "start": start}])
    assert len(store.list_events(settings)) == 1

    result = undo.undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone:")
    assert store.list_events(settings) == []


def test_undo_latest_reverts_reschedule(settings) -> None:
    start = _iso_in(settings, days=2)
    _apply(settings, [{"op": "create", "title": "Dentist", "start": start}])
    event_id = store.list_events(settings)[0].id

    new_start = _iso_in(settings, days=2, hours=1)
    _apply(settings, [{"op": "reschedule", "id": event_id, "start": new_start}])
    assert store.get_event(settings, event_id).start == new_start

    result = undo.undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone:")
    assert store.get_event(settings, event_id).start == start


def test_undo_latest_reverts_cancel_by_recreating(settings) -> None:
    event = store.create_event(settings, title="Dentist", start=_iso_in(settings, days=2))
    _apply(settings, [{"op": "cancel", "id": event.id}])
    assert store.get_event(settings, event.id) is None

    result = undo.undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone:")
    back = store.get_event(settings, event.id)
    assert back is not None and back == event


def test_undo_latest_reverts_skip_occurrence(settings) -> None:
    series = _weekly_series(settings)
    current = context.now(settings)
    from assistant.calendar import recurrence

    mondays = recurrence.occurrences_in(settings, current, current + timedelta(days=28))
    first = store.parse_dt(mondays[0].start)

    _apply(settings, [{"op": "skip", "id": series.id, "occurrence": first.isoformat()}])
    assert store.load_exdates(store.get_event(settings, series.id)) == [first.isoformat()]

    result = undo.undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone:")
    assert store.load_exdates(store.get_event(settings, series.id)) == []


def test_undo_latest_reverts_move_occurrence(settings) -> None:
    series = _weekly_series(settings)
    current = context.now(settings)
    from assistant.calendar import recurrence

    mondays = recurrence.occurrences_in(settings, current, current + timedelta(days=28))
    first = store.parse_dt(mondays[0].start)
    new_start = (first + timedelta(days=1, hours=1)).isoformat()

    _apply(
        settings,
        [{"op": "move", "id": series.id, "occurrence": first.isoformat(), "start": new_start}],
    )
    assert store.load_overrides(store.get_event(settings, series.id))

    result = undo.undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone:")
    assert store.load_overrides(store.get_event(settings, series.id)) == {}


def test_undo_latest_reverts_whole_batch(settings) -> None:
    keep_cancelled = store.create_event(
        settings, title="Yoga", start=_iso_in(settings, days=3)
    )
    applied = _apply(
        settings,
        [
            {"op": "create", "title": "Dentist", "start": _iso_in(settings, days=2)},
            {"op": "cancel", "id": keep_cancelled.id},
        ],
    )
    assert len(applied) == 2
    titles = {e.title for e in store.list_events(settings)}
    assert titles == {"Dentist"}  # Yoga cancelled, Dentist created

    result = undo.undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone:")
    titles = {e.title for e in store.list_events(settings)}
    assert titles == {"Yoga"}  # both reverted: Dentist gone, Yoga back


def test_undo_latest_only_targets_own_thread(settings) -> None:
    start = _iso_in(settings, days=2)
    _apply(settings, [{"op": "create", "title": "Dentist", "start": start}])

    assert undo.undo_latest(settings, "telegram:999", 15) == "Nothing to undo."
    assert len(store.list_events(settings)) == 1  # untouched


def test_undo_latest_is_idempotent_after_success(settings) -> None:
    start = _iso_in(settings, days=2)
    _apply(settings, [{"op": "create", "title": "Dentist", "start": start}])

    assert undo.undo_latest(settings, THREAD, 15).startswith("Undone:")
    assert undo.undo_latest(settings, THREAD, 15) == "Nothing to undo."


def test_undo_latest_respects_window(settings) -> None:
    start = _iso_in(settings, days=2)
    _apply(settings, [{"op": "create", "title": "Dentist", "start": start}])

    # Backdate the ledger entry well outside the window (avoids a same-second race
    # against real time, since applied_at only has second precision).
    stale = (context.now(settings) - timedelta(minutes=30)).isoformat(timespec="seconds")
    with undo._connect(settings) as conn:
        conn.execute("UPDATE write_log SET applied_at = ?", (stale,))

    assert undo.undo_latest(settings, THREAD, 15) == "Nothing recent enough to undo."
    assert len(store.list_events(settings)) == 1  # write still in place


def test_write_log_empty_when_confirmation_disabled(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_write_confirmation=False,
    )
    start = _iso_in(settings, days=2)
    _apply(settings, [{"op": "create", "title": "Dentist", "start": start}])

    assert len(store.list_events(settings)) == 1  # write still applied
    assert _log_rows(settings) == []  # but nothing logged
    assert undo.undo_latest(settings, THREAD, 15) == "Nothing to undo."


def test_undo_window_survives_offset_change(settings) -> None:
    # A stamp written under another UTC offset (a DST flip) is lexically far from
    # the cutoff string even when the instant is minutes old; the window check
    # must compare instants, not strings.

    event = store.create_event(settings, title="Dentist", start=_iso_in(settings, days=1))
    applied_utc = (context.now(settings) - timedelta(minutes=5)).astimezone(UTC)
    with undo._connect(settings) as conn:
        conn.execute(
            "INSERT INTO write_log"
            " (thread_id, batch_id, event_id, op, summary, before_json, applied_at)"
            " VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (THREAD, "b1", event.id, "create", "created: Dentist",
             applied_utc.isoformat(timespec="seconds")),
        )
    result = undo.undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone:")
    assert store.get_event(settings, event.id) is None
