"""Habit-log table for the Postgres backend (twin of assistant.habits.store)."""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect

_COLS = "id, habit, value, unit, note, logged_on, created"


def ensure_habits_schema(settings: Settings) -> None:
    if _schema_done(settings, "habits"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_habit_log (
              id TEXT PRIMARY KEY,
              habit TEXT NOT NULL,
              value DOUBLE PRECISION NOT NULL DEFAULT 0,
              unit TEXT NOT NULL DEFAULT '',
              note TEXT NOT NULL DEFAULT '',
              logged_on TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "habits")


def _entry_from_row(row: dict):
    from ..habits.store import HabitEntry

    return HabitEntry(
        id=str(row["id"]),
        habit=str(row["habit"]),
        value=float(row.get("value") or 0.0),
        unit=str(row.get("unit") or ""),
        note=str(row.get("note") or ""),
        logged_on=str(row.get("logged_on") or ""),
        created=str(row.get("created") or ""),
    )


def log_habit_entry(
    settings: Settings, habit: str, value: float = 0.0, unit: str = "", note: str = "", on: str = ""
):
    import uuid

    from ..habits import store as habit_store

    ensure_habits_schema(settings)
    logged_on = habit_store.parse_date(on) or habit_store._today(settings)
    entry = habit_store.HabitEntry(
        id=uuid.uuid4().hex[:12],
        habit=habit.strip(),
        value=habit_store._coerce_value(value),
        unit=unit.strip(),
        note=note.strip(),
        logged_on=logged_on.isoformat(),
        created=habit_store._stamp_now(settings),
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_habit_log (id, habit, value, unit, note, logged_on, created)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (entry.id, entry.habit, entry.value, entry.unit, entry.note,
             entry.logged_on, entry.created),
        )
    return entry


def get_habit_entry(settings: Settings, entry_id: str):
    ensure_habits_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(f"SELECT {_COLS} FROM assistant_habit_log WHERE id = %s", (entry_id,))
        )
    return _entry_from_row(rows[0]) if rows else None


def list_habit_entries(settings: Settings):
    ensure_habits_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_habit_log"))
    return [_entry_from_row(r) for r in rows]


def delete_habit_entry(settings: Settings, entry_id: str):
    existing = get_habit_entry(settings, entry_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_habit_log WHERE id = %s", (entry_id,))
    return existing
