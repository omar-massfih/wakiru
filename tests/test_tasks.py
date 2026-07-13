"""Task subsystem tests — store CRUD, the reconciling write path, and undo.

Everything runs for real (plain SQLite + stdlib datetime); only the Codex call in
the write path (via ops.update_tasks) and the outbound confirmation are faked,
matching test_calendar.py / test_undo.py.
"""

from __future__ import annotations

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
        enable_auto_tasks=True,
        enable_write_confirmation=True,
        write_undo_window_minutes=15,
    )


@pytest.fixture(autouse=True)
def _no_push(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.notify.deliver_write_confirmation", lambda *a, **k: True
    )


def _iso_in(settings: Settings, **delta) -> str:
    return (context.now(settings) + timedelta(**delta)).isoformat(timespec="minutes")


def _apply(settings: Settings, raw: str, monkeypatch, user="do it", reply="ok") -> list[str]:
    monkeypatch.setattr("assistant.tasks.ops.complete_text", lambda *a, **k: raw)
    return ops.update_tasks(settings, user, reply, THREAD)


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


# --- write path (ops.update_tasks) ------------------------------------------ #


def test_add_op_creates_task(settings, monkeypatch) -> None:
    applied = _apply(settings, '[{"op": "add", "title": "buy milk"}]', monkeypatch)
    assert applied == ["added task: buy milk"]
    assert [t.title for t in store.list_tasks(settings)] == ["buy milk"]


def test_complete_op_by_id(settings, monkeypatch) -> None:
    t = store.create_task(settings, "buy milk")
    applied = _apply(settings, f'[{{"op": "complete", "id": "{t.id}"}}]', monkeypatch)
    assert applied == ["completed: buy milk"]
    assert store.list_tasks(settings) == []


def test_remove_op_by_fuzzy_title(settings, monkeypatch) -> None:
    store.create_task(settings, "buy milk")
    applied = _apply(settings, '[{"op": "remove", "title": "milk"}]', monkeypatch)
    assert applied == ["removed task: buy milk"]
    assert store.list_tasks(settings) == []


def test_ambiguous_target_is_skipped(settings, monkeypatch) -> None:
    store.create_task(settings, "buy milk")
    store.create_task(settings, "buy bread")
    applied = _apply(settings, '[{"op": "remove", "title": "buy"}]', monkeypatch)
    assert applied == []  # ambiguous — reverting nothing beats removing the wrong one
    assert len(store.list_tasks(settings)) == 2


def test_update_op_never_targets_by_its_new_title(settings, monkeypatch) -> None:
    # For an "update" the schema's `title` is the REPLACEMENT value. Using it to
    # look the target up resolves to whichever task already bears that title —
    # a row the user never named.
    renamed = store.create_task(settings, "buy milk")
    bystander = store.create_task(settings, "buy milk and eggs")

    applied = _apply(
        settings,
        '[{"op": "update", "title": "buy milk and eggs", "notes": "corner shop"}]',
        monkeypatch,
    )

    assert applied == []  # no id and no query, so there is nothing to target
    assert store.get_task(settings, bystander.id).notes == ""
    assert store.get_task(settings, renamed.id).title == "buy milk"


def test_update_op_by_id_still_renames(settings, monkeypatch) -> None:
    task = store.create_task(settings, "buy milk")
    applied = _apply(
        settings,
        f'[{{"op": "update", "id": "{task.id}", "title": "buy milk and eggs"}}]',
        monkeypatch,
    )
    assert applied == ["updated task: buy milk and eggs"]


def test_auto_tasks_disabled_is_noop(tmp_path, monkeypatch) -> None:
    settings = Settings(memory_dir=str(tmp_path / "m"), enable_auto_tasks=False)
    applied = _apply(settings, '[{"op": "add", "title": "x"}]', monkeypatch)
    assert applied == []


def test_context_flags_overdue(settings) -> None:
    store.create_task(settings, "overdue thing", due=_iso_in(settings, days=-1))
    assert "OVERDUE" in tasks_context(settings)


# --- undo (via the cross-ledger arbiter) ------------------------------------ #


def test_undo_reverts_add(settings, monkeypatch) -> None:
    _apply(settings, '[{"op": "add", "title": "buy milk"}]', monkeypatch)
    result = undo_latest(settings, THREAD, 15)
    assert result == "Undone: removed: buy milk"
    assert store.list_tasks(settings) == []


def test_undo_restores_completed_task(settings, monkeypatch) -> None:
    t = store.create_task(settings, "buy milk")
    _apply(settings, f'[{{"op": "complete", "id": "{t.id}"}}]', monkeypatch)
    assert store.list_tasks(settings) == []  # completed
    result = undo_latest(settings, THREAD, 15)
    assert result.startswith("Undone: restored:")
    assert [x.title for x in store.list_tasks(settings)] == ["buy milk"]  # open again


def test_undo_restores_removed_task(settings, monkeypatch) -> None:
    store.create_task(settings, "buy milk")
    _apply(settings, '[{"op": "remove", "title": "buy milk"}]', monkeypatch)
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
    # Exact minutes drift with wall-clock; just assert the phrasing shape.
    assert fired[0]["message"].startswith("Task due: submit form in ")
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
    assert messages == [
        "Task due: submit form in 15 min",
        "Task due: submit form now",
        "Task overdue: submit form (15 min ago)",
        "Task overdue: submit form (30 min ago)",
    ]


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


def test_undo_arbiter_reverts_most_recent_across_ledgers(settings, monkeypatch) -> None:
    import time

    from assistant.calendar import ops as cal_ops
    from assistant.calendar import store as cal_store

    # An event write, then (a moment later) a task write on the same thread.
    start = _iso_in(settings, days=2)
    monkeypatch.setattr(
        "assistant.calendar.ops.complete_text",
        lambda *a, **k: f'[{{"op": "create", "title": "Dentist", "start": "{start}"}}]',
    )
    cal_ops.update_calendar(settings, "book dentist", "Done", THREAD)
    time.sleep(1.1)  # seconds-precision stamps must differ so the arbiter can order them
    _apply(settings, '[{"op": "add", "title": "buy milk"}]', monkeypatch)

    # The task was written most recently, so "undo" reverts it and leaves the event.
    assert undo_latest(settings, THREAD, 15) == "Undone: removed: buy milk"
    assert [t.title for t in store.list_tasks(settings)] == []
    assert [e.title for e in cal_store.list_events(settings)] == ["Dentist"]
