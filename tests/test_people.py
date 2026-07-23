"""People subsystem tests — store CRUD, the tool write path, undo, and the
read-path attention signals (overdue contact, upcoming birthday).

Everything runs for real (plain SQLite + stdlib datetime); writes are exercised
by applying parsed operations directly through ``ops.apply_op`` — exactly what
the people tools do — matching test_tasks.py / test_calendar.py.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from assistant.calendar import context as cal_context
from assistant.config import Settings
from assistant.people import ops, store
from assistant.people.context import (
    attention_lines,
    briefing_people,
    days_until_birthday,
    is_overdue,
    people_context,
)
from assistant.people.reminders import run_birthday_reminders
from assistant.undo import undo_latest

THREAD = "telegram:1"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_people=True,
        enable_write_confirmation=True,
        write_undo_window_minutes=15,
        people_birthday_lead_days=7,
    )


def _apply(settings: Settings, ops_list: list[dict]) -> list[str]:
    """Apply parsed operations as the people tools do — one undo batch per call."""
    batch_id = uuid.uuid4().hex
    applied = []
    for op in ops_list:
        result = ops.apply_op(settings, op, THREAD, batch_id)
        if result:
            applied.append(result)
    return applied


# --- store CRUD ------------------------------------------------------------- #


def test_create_and_list(settings) -> None:
    store.create_person(settings, "Kari", relationship="sister", cadence_days=14)
    people = store.list_people(settings)
    assert [p.name for p in people] == ["Kari"]
    assert people[0].relationship == "sister"
    assert people[0].cadence_days == 14


def test_update_and_delete(settings) -> None:
    p = store.create_person(settings, "Ola")
    store.update_person(settings, p.id, relationship="colleague", notes="likes tea")
    assert store.get_person(settings, p.id).relationship == "colleague"
    assert store.delete_person(settings, p.id) is not None
    assert store.get_person(settings, p.id) is None


def test_log_contact_stamps_last_contact(settings) -> None:
    p = store.create_person(settings, "Ola", cadence_days=7)
    assert p.last_contact == ""
    updated = store.log_contact(settings, p.id)
    assert updated.last_contact != ""


def test_find_by_name_and_id(settings) -> None:
    p = store.create_person(settings, "Kari Nordmann")
    assert store.find_person(settings, p.id).id == p.id
    assert store.find_person(settings, "kari").id == p.id
    assert store.find_person(settings, "nobody") is None


# --- tool write path -------------------------------------------------------- #


def test_add_op_and_dedupe(settings) -> None:
    assert _apply(settings, [{"op": "add", "name": "Kari", "relationship": "sister"}])
    # A second add with the same name is refused, not duplicated.
    result = ops.apply_op(settings, {"op": "add", "name": "Kari"}, THREAD, uuid.uuid4().hex)
    assert "already exists" in result
    assert len(store.list_people(settings)) == 1


def test_update_op_by_name(settings) -> None:
    _apply(settings, [{"op": "add", "name": "Ola"}])
    _apply(settings, [{"op": "update", "query": "Ola", "cadence_days": "30"}])
    assert store.find_person(settings, "Ola").cadence_days == 30


def test_log_contact_op(settings) -> None:
    _apply(settings, [{"op": "add", "name": "Ola", "cadence_days": "7"}])
    result = _apply(settings, [{"op": "log_contact", "query": "Ola"}])
    assert result and "logged contact" in result[0]
    assert store.find_person(settings, "Ola").last_contact != ""


def test_ambiguous_reference(settings) -> None:
    store.create_person(settings, "Kari Nordmann")
    store.create_person(settings, "Kari Hansen")
    result = ops.apply_op(settings, {"op": "update", "query": "Kari", "notes": "x"}, THREAD, uuid.uuid4().hex)
    assert "Ambiguous" in result


def test_remove_op(settings) -> None:
    _apply(settings, [{"op": "add", "name": "Ola"}])
    _apply(settings, [{"op": "remove", "query": "Ola"}])
    assert store.list_people(settings) == []


# --- undo ------------------------------------------------------------------- #


def test_undo_reverts_add(settings) -> None:
    _apply(settings, [{"op": "add", "name": "Kari"}])
    assert len(store.list_people(settings)) == 1
    message = undo_latest(settings, THREAD, 15)
    assert "removed: Kari" in message
    assert store.list_people(settings) == []


def test_undo_restores_after_remove(settings) -> None:
    _apply(settings, [{"op": "add", "name": "Ola", "relationship": "friend"}])
    _apply(settings, [{"op": "remove", "query": "Ola"}])
    assert store.list_people(settings) == []
    undo_latest(settings, THREAD, 15)
    restored = store.find_person(settings, "Ola")
    assert restored is not None and restored.relationship == "friend"


# --- read path: attention signals ------------------------------------------- #


def test_overdue_contact_flagged(settings) -> None:
    p = store.create_person(settings, "Kari", relationship="sister", cadence_days=7)
    long_ago = (cal_context.now(settings) - timedelta(days=20)).isoformat()
    store.update_person(settings, p.id, last_contact=long_ago)
    current = cal_context.now(settings)
    assert is_overdue(store.get_person(settings, p.id), current) is True
    block = people_context(settings)
    assert "overdue" in block
    assert any("overdue" in line for line in attention_lines(settings, current))


def test_birthday_soon_flagged(settings) -> None:
    today = cal_context.now(settings).date()
    store.create_person(settings, "Ola", birthday=f"{today.month:02d}-{today.day:02d}")
    current = cal_context.now(settings)
    p = store.find_person(settings, "Ola")
    assert days_until_birthday(p, current) == 0
    assert "🎂" in people_context(settings)
    assert "🎂" in briefing_people(settings)


def test_no_attention_when_nothing_due(settings) -> None:
    store.create_person(settings, "Ola")  # no cadence, no birthday
    assert attention_lines(settings, cal_context.now(settings)) == []
    assert briefing_people(settings) == ""


# --- birthday reminders ----------------------------------------------------- #


@pytest.fixture
def _birthday_delivery(monkeypatch):
    """Capture pushed birthday reminders; model composer returns the fallback."""
    sent: list[dict] = []
    monkeypatch.setattr("assistant.compose.compose_push", lambda s, **kw: kw["fallback"])
    monkeypatch.setattr(
        "assistant.people.reminders.deliver_reminder",
        lambda s, r: bool(sent.append(r)) or True,
    )
    monkeypatch.setattr("assistant.proactive.record_push", lambda *a, **k: None)
    return sent


def _birthday_on(settings, name, offset_days) -> None:
    date = cal_context.now(settings).date() + timedelta(days=offset_days)
    store.create_person(settings, name, birthday=f"{date.month:02d}-{date.day:02d}")


def test_birthday_reminder_fires_once(settings, _birthday_delivery) -> None:
    _birthday_on(settings, "Kari", 0)  # birthday today
    sent = run_birthday_reminders(settings)
    assert len(sent) == 1
    assert "Kari" in _birthday_delivery[0]["message"]
    # Exactly-once: a second pass claims nothing.
    assert run_birthday_reminders(settings) == []


def test_birthday_within_lead_window(settings, _birthday_delivery) -> None:
    _birthday_on(settings, "Ola", settings.people_birthday_lead_days)  # edge of window
    assert len(run_birthday_reminders(settings)) == 1


def test_birthday_outside_lead_window(settings, _birthday_delivery) -> None:
    _birthday_on(settings, "Ola", settings.people_birthday_lead_days + 5)
    assert run_birthday_reminders(settings) == []


def test_birthday_reminders_respect_switches(settings, _birthday_delivery) -> None:
    _birthday_on(settings, "Kari", 0)
    assert run_birthday_reminders(settings.model_copy(update={"enable_reminders": False})) == []
    assert run_birthday_reminders(settings.model_copy(update={"enable_people": False})) == []
