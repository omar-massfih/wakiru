"""Task subsystem tests — store CRUD, the tool write path, and undo.

Everything runs for real (plain SQLite + stdlib datetime); writes are exercised
by applying parsed operations directly through ``ops.apply_op`` — exactly what
the task tools do — matching test_calendar.py / test_undo.py.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from assistant.calendar import context
from assistant.config import Settings
from assistant.tasks import ops, reminders, store
from assistant.tasks.context import tasks_context
from assistant.undo import undo_latest

THREAD = "telegram:1"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_tasks=True,
        enable_write_confirmation=True,
        write_undo_window_minutes=15,
    )


@pytest.fixture(autouse=True)
def _compose_fallback(monkeypatch) -> None:
    """Stand-in composer: behaves like a failed model (returns the fallback)."""
    monkeypatch.setattr(
        "assistant.compose.compose_push", lambda s, **kw: kw["fallback"]
    )


def _iso_in(settings: Settings, **delta) -> str:
    return (context.now(settings) + timedelta(**delta)).isoformat(timespec="minutes")


def _apply(settings: Settings, ops_list: list[dict]) -> list[str]:
    """Apply parsed operations as the task tools do — one undo batch per call."""
    batch_id = uuid.uuid4().hex
    applied = []
    for op in ops_list:
        result = ops.apply_op(settings, op, THREAD, batch_id)
        if result:
            applied.append(result)
    return applied


# --- store CRUD ------------------------------------------------------------- #


def test_create_and_list_open(settings) -> None:
    store.create_task(settings, "call plumber")
    open_tasks = store.list_tasks(settings)
    assert [t.title for t in open_tasks] == ["call plumber"]
    assert open_tasks[0].done is False


def test_complete_moves_out_of_open_and_sets_done_at(settings) -> None:
    t = store.create_task(settings, "buy milk")
    done = store.complete_task(settings, t.id)
    assert done.done is True and done.done_at
    assert store.list_tasks(settings) == []
    assert [x.title for x in store.list_tasks(settings, include_done=True)] == ["buy milk"]


def test_complete_is_idempotent(settings) -> None:
    t = store.create_task(settings, "x")
    first = store.complete_task(settings, t.id)
    second = store.complete_task(settings, t.id)
    assert first.done_at == second.done_at  # not re-stamped


def test_open_tasks_sort_dated_before_undated(settings) -> None:
    store.create_task(settings, "undated")
    store.create_task(settings, "later", due=_iso_in(settings, days=5))
    store.create_task(settings, "soon", due=_iso_in(settings, days=1))
    assert [t.title for t in store.list_tasks(settings)] == ["soon", "later", "undated"]


def test_find_task_prefers_open_over_done(settings) -> None:
    done = store.create_task(settings, "review report")
    store.complete_task(settings, done.id)
    open_one = store.create_task(settings, "review report")  # same title, still open
    assert store.find_task(settings, "review report").id == open_one.id


def test_update_task_fields(settings) -> None:
    t = store.create_task(settings, "draft email")
    revised = store.update_task(settings, t.id, title="draft the email", notes="to Bob")
    assert revised.title == "draft the email" and revised.notes == "to Bob"


def test_due_naive_gets_timezone(settings) -> None:
    t = store.create_task(settings, "pay rent", due="2999-01-01T09:00:00")
    # A naive due gets the assistant's offset attached on the way in.
    assert "+" in t.due or t.due.endswith("Z")


# --- write path (ops.apply_op) ----------------------------------------------- #


def test_add_op_creates_task(settings) -> None:
    applied = _apply(settings, [{"op": "add", "title": "buy milk"}])
    assert applied == ["added task: buy milk"]
    assert [t.title for t in store.list_tasks(settings)] == ["buy milk"]


def test_complete_op_by_id(settings) -> None:
    t = store.create_task(settings, "buy milk")
    applied = _apply(settings, [{"op": "complete", "id": t.id}])
    assert applied == ["completed: buy milk"]
    assert store.list_tasks(settings) == []


def test_remove_op_by_fuzzy_title(settings) -> None:
    store.create_task(settings, "buy milk")
    applied = _apply(settings, [{"op": "remove", "title": "milk"}])
    assert applied == ["removed task: buy milk"]
    assert store.list_tasks(settings) == []


def test_ambiguous_target_is_skipped(settings) -> None:
    store.create_task(settings, "buy milk")
    store.create_task(settings, "buy bread")
    applied = _apply(settings, [{"op": "remove", "title": "buy"}])
    # ambiguous — reverting nothing beats removing the wrong one, but the
    # model needs to know *which* candidates collided so it can retry by id.
    assert len(applied) == 1
    assert "Ambiguous" in applied[0]
    assert "buy milk" in applied[0] and "buy bread" in applied[0]
    assert len(store.list_tasks(settings)) == 2


def test_complete_op_ambiguous_returns_candidates(settings) -> None:
    a = store.create_task(settings, "water plants")
    b = store.create_task(settings, "water plants", due=_iso_in(settings, days=1))
    applied = _apply(settings, [{"op": "complete", "title": "water plants"}])
    assert len(applied) == 1
    assert "Ambiguous" in applied[0]
    assert a.id in applied[0] and b.id in applied[0]
    assert len(store.list_tasks(settings)) == 2


def test_add_op_refuses_exact_title_duplicate(settings) -> None:
    original = store.create_task(settings, "buy milk")
    applied = _apply(settings, [{"op": "add", "title": "  Buy Milk  "}])
    assert len(applied) == 1
    assert applied[0].startswith("Not added")
    assert original.id in applied[0]
    assert [t.title for t in store.list_tasks(settings)] == ["buy milk"]


def test_add_op_allows_distinct_title(settings) -> None:
    store.create_task(settings, "Buy milk")
    applied = _apply(settings, [{"op": "add", "title": "Buy milk and eggs"}])
    assert applied == ["added task: Buy milk and eggs"]
    assert {t.title for t in store.list_tasks(settings)} == {"Buy milk", "Buy milk and eggs"}


def test_add_op_dedupe_ignores_completed_tasks(settings) -> None:
    t = store.create_task(settings, "buy milk")
    store.complete_task(settings, t.id)
    applied = _apply(settings, [{"op": "add", "title": "buy milk"}])
    assert applied == ["added task: buy milk"]
    assert [t.title for t in store.list_tasks(settings)] == ["buy milk"]


def test_update_op_never_targets_by_its_new_title(settings) -> None:
    # For an "update" the schema's `title` is the REPLACEMENT value. Using it to
    # look the target up resolves to whichever task already bears that title —
    # a row the user never named.
    renamed = store.create_task(settings, "buy milk")
    bystander = store.create_task(settings, "buy milk and eggs")

    applied = _apply(
        settings, [{"op": "update", "title": "buy milk and eggs", "notes": "corner shop"}]
    )

    assert applied == []  # no id and no query, so there is nothing to target
    assert store.get_task(settings, bystander.id).notes == ""
    assert store.get_task(settings, renamed.id).title == "buy milk"


def test_update_op_by_id_still_renames(settings) -> None:
    task = store.create_task(settings, "buy milk")
    applied = _apply(settings, [{"op": "update", "id": task.id, "title": "buy milk and eggs"}])
    assert applied == ["updated task: buy milk and eggs"]


def test_context_flags_overdue(settings) -> None:
    store.create_task(settings, "overdue thing", due=_iso_in(settings, days=-1))
    assert "OVERDUE" in tasks_context(settings)


# --- recurring tasks (rrule roll-forward) ----------------------------------- #


def test_complete_recurring_rolls_due_forward(settings) -> None:
    t = store.create_task(
        settings, "water plants", due=_iso_in(settings, days=-1), rrule="FREQ=DAILY"
    )
    rolled = store.complete_task(settings, t.id)
    assert rolled.done is False and rolled.done_at == ""  # still open
    due = store.parse_dt(rolled.due)
    assert due is not None and due > context.now(settings)  # next occurrence
    assert rolled.rrule == "FREQ=DAILY"  # rule survives the roll


def test_complete_recurring_keeps_wall_clock_time(settings) -> None:
    anchor = (context.now(settings) - timedelta(days=3)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    t = store.create_task(
        settings, "standup notes", due=anchor.isoformat(), rrule="FREQ=DAILY"
    )
    rolled = store.complete_task(settings, t.id)
    due = store.parse_dt(rolled.due)
    assert (due.hour, due.minute) == (9, 0)


def test_complete_exhausted_rule_closes_normally(settings) -> None:
    t = store.create_task(
        settings, "one-off", due=_iso_in(settings, days=-1), rrule="FREQ=DAILY;COUNT=1"
    )
    done = store.complete_task(settings, t.id)
    assert done.done is True and done.done_at  # no next occurrence: closed


def test_add_op_drops_rule_without_due_anchor(settings) -> None:
    applied = _apply(settings, [{"op": "add", "title": "floating chore", "rrule": "FREQ=DAILY"}])
    assert store.list_tasks(settings)[0].rrule == ""
    assert "recurrence ignored" in applied[0]  # the model is told, not misled


def test_add_op_drops_rule_on_unparseable_due(settings) -> None:
    applied = _apply(settings, [{
        "op": "add", "title": "trash", "due": "next Friday", "rrule": "FREQ=WEEKLY",
    }])
    assert store.list_tasks(settings)[0].rrule == ""  # a rule that could never roll
    assert "recurrence ignored" in applied[0]


def test_add_op_drops_unparseable_rule(settings) -> None:
    applied = _apply(settings, [{
        "op": "add", "title": "vague chore",
        "due": _iso_in(settings, days=1), "rrule": "sometimes",
    }])
    assert store.list_tasks(settings)[0].rrule == ""
    assert "recurrence ignored" in applied[0]


def test_add_op_humanizes_recurring_summary(settings) -> None:
    applied = _apply(settings, [{
        "op": "add", "title": "take out trash",
        "due": _iso_in(settings, days=1), "rrule": "FREQ=WEEKLY",
    }])
    assert applied == ["added task: take out trash (weekly)"]


def test_complete_op_reports_next_due(settings) -> None:
    t = store.create_task(
        settings, "water plants", due=_iso_in(settings, days=-1), rrule="FREQ=WEEKLY"
    )
    applied = _apply(settings, [{"op": "complete", "id": t.id}])
    assert applied and applied[0].startswith("completed: water plants (recurs — next due")


def test_update_op_ignores_unparseable_rule(settings) -> None:
    t = store.create_task(
        settings, "water plants", due=_iso_in(settings, days=1), rrule="FREQ=WEEKLY"
    )
    applied = _apply(settings, [{"op": "update", "id": t.id, "rrule": "whenever"}])
    assert store.get_task(settings, t.id).rrule == "FREQ=WEEKLY"  # untouched
    assert "recurrence ignored" in applied[0]


def test_update_op_ignores_rule_on_undated_task(settings) -> None:
    t = store.create_task(settings, "floating chore")
    _apply(settings, [{"op": "update", "id": t.id, "rrule": "FREQ=DAILY"}])
    assert store.get_task(settings, t.id).rrule == ""  # no due to anchor


def test_update_op_accepts_rule_with_new_due_in_same_op(settings) -> None:
    t = store.create_task(settings, "floating chore")
    _apply(settings, [{
        "op": "update", "id": t.id,
        "due": _iso_in(settings, days=1), "rrule": "FREQ=DAILY",
    }])
    assert store.get_task(settings, t.id).rrule == "FREQ=DAILY"


def test_doubled_complete_in_one_turn_rolls_once(settings) -> None:
    t = store.create_task(
        settings, "water plants", due=_iso_in(settings, days=-1), rrule="FREQ=DAILY"
    )
    applied = _apply(settings, [
        {"op": "complete", "id": t.id},
        {"op": "complete", "id": t.id},  # duplicate tool call in the same turn
    ])
    assert applied[1] == "already completed this turn: water plants"
    first_roll = store.get_task(settings, t.id).due
    # A later turn (fresh batch) may complete again — and rolls again.
    _apply(settings, [{"op": "complete", "id": t.id}])
    assert store.get_task(settings, t.id).due != first_roll


def test_undo_restores_rolled_due(settings) -> None:
    original_due = _iso_in(settings, days=-1)
    t = store.create_task(settings, "water plants", due=original_due, rrule="FREQ=DAILY")
    stored_due = store.get_task(settings, t.id).due
    _apply(settings, [{"op": "complete", "id": t.id}])
    assert store.get_task(settings, t.id).due != stored_due  # rolled
    result = undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone: restored:")
    assert store.get_task(settings, t.id).due == stored_due  # back on the old due


def test_context_shows_recurrence(settings) -> None:
    store.create_task(
        settings, "water plants", due=_iso_in(settings, days=1), rrule="FREQ=WEEKLY"
    )
    assert "repeats weekly" in tasks_context(settings)


def test_store_migrates_pre_rrule_db(settings, tmp_path) -> None:
    # A DB created before the rrule column existed gains it in place.
    import sqlite3 as sqlite3_mod
    from pathlib import Path

    Path(settings.tasks_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3_mod.connect(settings.tasks_db_path)
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL,"
        " done INTEGER DEFAULT 0, due TEXT DEFAULT '', notes TEXT DEFAULT '',"
        " created TEXT DEFAULT '', updated TEXT DEFAULT '', done_at TEXT DEFAULT '')"
    )
    conn.execute("INSERT INTO tasks (id, title) VALUES ('old1', 'legacy task')")
    conn.commit()
    conn.close()
    tasks = store.list_tasks(settings)
    assert [t.title for t in tasks] == ["legacy task"]
    assert tasks[0].rrule == ""


# --- undo (via the cross-ledger arbiter) ------------------------------------ #


def test_undo_reverts_add(settings) -> None:
    _apply(settings, [{"op": "add", "title": "buy milk"}])
    result = undo_latest(settings, THREAD, 15)
    assert result == "Undone: removed: buy milk"
    assert store.list_tasks(settings) == []


def test_undo_restores_completed_task(settings) -> None:
    t = store.create_task(settings, "buy milk")
    _apply(settings, [{"op": "complete", "id": t.id}])
    assert store.list_tasks(settings) == []  # completed
    result = undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone: restored:")
    assert [x.title for x in store.list_tasks(settings)] == ["buy milk"]  # open again


def test_undo_restores_removed_task(settings) -> None:
    store.create_task(settings, "buy milk")
    _apply(settings, [{"op": "remove", "title": "buy milk"}])
    undo_latest(settings, THREAD, 15)
    assert [x.title for x in store.list_tasks(settings)] == ["buy milk"]


def test_undo_nothing_recent(settings) -> None:
    assert undo_latest(settings, THREAD, 15) == "Nothing to undo."


# --- due-task reminders ----------------------------------------------------- #


def test_reminder_fires_within_lead(settings) -> None:
    settings = settings.model_copy(update={"reminder_lead_minutes": [60]})
    store.create_task(settings, "submit form", due=_iso_in(settings, minutes=30))
    fired = reminders.run_task_reminders(settings)
    assert len(fired) == 1
    assert fired[0]["title"] == "submit form"
    # Exact minutes drift with wall-clock; just assert the essentials.
    assert "submit form" in fired[0]["message"]
    assert "min" in fired[0]["message"]


def test_reminder_outside_lead_not_fired(settings) -> None:
    settings = settings.model_copy(update={"reminder_lead_minutes": [60]})
    store.create_task(settings, "later", due=_iso_in(settings, hours=5))
    assert reminders.run_task_reminders(settings) == []


def test_reminder_fires_once_then_deduped(settings) -> None:
    settings = settings.model_copy(update={"reminder_lead_minutes": [60]})
    store.create_task(settings, "submit form", due=_iso_in(settings, minutes=30))
    assert len(reminders.run_task_reminders(settings)) == 1
    assert reminders.run_task_reminders(settings) == []  # already claimed


def test_reminder_ignores_undated_and_done(settings) -> None:
    settings = settings.model_copy(update={"reminder_lead_minutes": [60]})
    store.create_task(settings, "undated")  # no due
    done = store.create_task(settings, "done soon", due=_iso_in(settings, minutes=10))
    store.complete_task(settings, done.id)  # completed → not open
    assert reminders.run_task_reminders(settings) == []


def test_repeat_nags_overdue_then_stops(settings, monkeypatch) -> None:
    settings = settings.model_copy(
        update={
            "reminder_lead_minutes": [60],
            "reminder_repeat_minutes": 15,
            "reminder_overdue_max_minutes": 30,
        }
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    due = (base + timedelta(minutes=15)).isoformat(timespec="seconds")
    store.create_task(settings, "submit form", due=due)

    messages: list[str] = []
    for step in range(0, 61, 15):
        monkeypatch.setattr(reminders, "now", lambda s, t=base + timedelta(minutes=step): t)
        messages += [r["message"] for r in reminders.run_task_reminders(settings)]

    # Nudges at due-15, due, then overdue every 15 min up to the 30-min window;
    # the due+45 step is past reminder_overdue_max_minutes, so it stays silent.
    assert len(messages) == 4
    assert all("submit form" in m for m in messages)
    for m, countdown in zip(messages, ["15 min", "now", "15 min ago", "30 min ago"], strict=True):
        assert countdown in m


def test_repeat_overdue_capped_by_nudge_count(settings, monkeypatch) -> None:
    # A generous time window, but a 2-nudge count cap → overdue nags stop after
    # two, even though reminder_overdue_max_minutes would allow far more.
    settings = settings.model_copy(
        update={
            "reminder_lead_minutes": [60],
            "reminder_repeat_minutes": 15,
            "reminder_overdue_max_minutes": 1440,
            "reminder_overdue_max_nudges": 2,
        }
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    store.create_task(settings, "call bank", due=base.isoformat(timespec="seconds"))

    overdue: list[str] = []
    for step in (15, 30, 45):  # 15, 30, then 45 min past due
        monkeypatch.setattr(reminders, "now", lambda s, t=base + timedelta(minutes=step): t)
        overdue += [r["message"] for r in reminders.run_task_reminders(settings)]
    # Cap = 2 * 15 min = 30 min, so +15 and +30 fire but +45 is silent.
    assert len(overdue) == 2


def test_notify_only_flag_round_trips_and_clears(settings) -> None:
    t = store.create_task(settings, "session resets", notify_only=True)
    assert store.get_task(settings, t.id).notify_only is True
    store.update_task(settings, t.id, notify_only="false")
    assert store.get_task(settings, t.id).notify_only is False


def test_notify_only_reminder_does_not_nag_overdue(settings, monkeypatch) -> None:
    settings = settings.model_copy(
        update={
            "reminder_lead_minutes": [60],
            "reminder_repeat_minutes": 15,
            "reminder_overdue_max_minutes": 120,
        }
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    _apply(
        settings,
        [{"op": "add", "title": "session resets", "notify_only": "true",
          "due": base.isoformat(timespec="seconds")}],
    )
    assert store.find_task(settings, "session resets").notify_only is True

    messages: list[str] = []
    for step in (0, 15, 30):  # at due, then 15 and 30 min overdue
        monkeypatch.setattr(reminders, "now", lambda s, t=base + timedelta(minutes=step): t)
        messages += [r["message"] for r in reminders.run_task_reminders(settings)]
    # Fires once at its time, then stays quiet — a normal repeat-mode task would
    # keep nagging overdue here (see test_repeat_nags_overdue_then_stops).
    assert len(messages) == 1


def test_repeat_overdue_stops_once_done(settings, monkeypatch) -> None:
    settings = settings.model_copy(
        update={"reminder_lead_minutes": [60], "reminder_repeat_minutes": 15}
    )
    base = context.now(settings).replace(second=0, microsecond=0)
    task = store.create_task(
        settings, "submit form", due=(base + timedelta(minutes=15)).isoformat(timespec="seconds")
    )

    monkeypatch.setattr(reminders, "now", lambda s: base + timedelta(minutes=30))
    assert len(reminders.run_task_reminders(settings)) == 1  # overdue nag fires

    store.complete_task(settings, task.id)
    monkeypatch.setattr(reminders, "now", lambda s: base + timedelta(minutes=45))
    assert reminders.run_task_reminders(settings) == []  # done → no longer listed


def test_undo_arbiter_reverts_most_recent_across_ledgers(settings) -> None:
    import time

    from assistant.calendar import ops as cal_ops
    from assistant.calendar import store as cal_store

    # An event write, then (a moment later) a task write on the same thread.
    start = _iso_in(settings, days=2)
    cal_ops.apply_op(
        settings,
        {"op": "create", "title": "Dentist", "start": start},
        THREAD,
        uuid.uuid4().hex,
    )
    time.sleep(1.1)  # seconds-precision stamps must differ so the arbiter can order them
    _apply(settings, [{"op": "add", "title": "buy milk"}])

    # The task was written most recently, so "undo" reverts it and leaves the event.
    assert undo_latest(settings, THREAD, 15) == "Undone: removed: buy milk"
    assert [t.title for t in store.list_tasks(settings)] == []
    assert [e.title for e in cal_store.list_events(settings)] == ["Dentist"]
