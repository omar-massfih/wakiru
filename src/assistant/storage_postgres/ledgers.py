"""Write-undo ledgers and exactly-once reminder claims for the Postgres backend."""

from __future__ import annotations

from ..config import Settings
from .calendar import ensure_calendar_schema
from .core import (
    _executemany,
    _rows,
    connect,
)
from .tasks import ensure_tasks_schema


def record_calendar_write(settings: Settings, thread_id: str, batch_id: str, event_id: str, op: str, summary: str, before_json: str | None, applied_at: str) -> None:
    if not thread_id:
        return
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_calendar_write_log (thread_id, batch_id, event_id, op, summary, before_json, applied_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (thread_id, batch_id, event_id, op, summary, before_json, applied_at),
        )


def calendar_write_rows(settings: Settings, thread_id: str) -> list[dict]:
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        return _rows(conn.execute("SELECT * FROM assistant_calendar_write_log WHERE thread_id = %s AND undone_at IS NULL ORDER BY id DESC", (thread_id,)))


def mark_calendar_writes_undone(settings: Settings, ids: list[int], undone_at: str) -> None:
    if not ids:
        return
    ensure_calendar_schema(settings)
    with connect(settings) as conn:
        _executemany(conn, "UPDATE assistant_calendar_write_log SET undone_at = %s WHERE id = %s", [(undone_at, i) for i in ids])


def record_task_write(settings: Settings, thread_id: str, batch_id: str, task_id: str, op: str, summary: str, before_json: str | None, applied_at: str) -> None:
    if not thread_id:
        return
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_task_write_log (thread_id, batch_id, task_id, op, summary, before_json, applied_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (thread_id, batch_id, task_id, op, summary, before_json, applied_at),
        )


def task_write_rows(settings: Settings, thread_id: str) -> list[dict]:
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        return _rows(conn.execute("SELECT * FROM assistant_task_write_log WHERE thread_id = %s AND undone_at IS NULL ORDER BY id DESC", (thread_id,)))


def mark_task_writes_undone(settings: Settings, ids: list[int], undone_at: str) -> None:
    if not ids:
        return
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        _executemany(conn, "UPDATE assistant_task_write_log SET undone_at = %s WHERE id = %s", [(undone_at, i) for i in ids])


def claim_calendar_reminders(settings: Settings, reminders: list[dict], fired_at: str, current) -> list[dict]:
    from datetime import timedelta

    from ..calendar import store as calendar_store
    from ..fired_ledger import LEDGER_RETENTION_DAYS

    ensure_calendar_schema(settings)
    cutoff = current - timedelta(days=LEDGER_RETENTION_DAYS)
    sent: list[dict] = []
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT event_id, event_start, lead_minutes, fired_at FROM assistant_calendar_reminders_fired"))
        stale = [
            (r["event_id"], r["event_start"], r["lead_minutes"])
            for r in rows
            if (fired := calendar_store.parse_dt(str(r["fired_at"]))) is None or fired < cutoff
        ]
        _executemany(
            conn,
            "DELETE FROM assistant_calendar_reminders_fired WHERE event_id = %s AND event_start = %s AND lead_minutes = %s",
            stale,
        )
        for reminder in reminders:
            claimed = 0
            for lead in reminder["covered_leads"]:
                cur = conn.execute(
                    "INSERT INTO assistant_calendar_reminders_fired (event_id, event_start, lead_minutes, fired_at) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING event_id",
                    (reminder["event_id"], reminder["start"], lead, fired_at),
                )
                claimed += 1 if cur.fetchone() else 0
            if claimed:
                sent.append(reminder)
    return sent


def claim_task_reminders(settings: Settings, reminders: list[dict], fired_at: str, current) -> list[dict]:
    from datetime import timedelta

    from ..calendar.store import parse_dt
    from ..fired_ledger import LEDGER_RETENTION_DAYS

    ensure_tasks_schema(settings)
    cutoff = current - timedelta(days=LEDGER_RETENTION_DAYS)
    sent: list[dict] = []
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT task_id, due, lead_minutes, fired_at FROM assistant_task_reminders_fired"))
        stale = [
            (r["task_id"], r["due"], r["lead_minutes"])
            for r in rows
            if (fired := parse_dt(str(r["fired_at"]))) is None or fired < cutoff
        ]
        _executemany(
            conn,
            "DELETE FROM assistant_task_reminders_fired WHERE task_id = %s AND due = %s AND lead_minutes = %s",
            stale,
        )
        for reminder in reminders:
            claimed = 0
            for lead in reminder["covered_leads"]:
                cur = conn.execute(
                    "INSERT INTO assistant_task_reminders_fired (task_id, due, lead_minutes, fired_at) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING task_id",
                    (reminder["task_id"], reminder["due"], lead, fired_at),
                )
                claimed += 1 if cur.fetchone() else 0
            if claimed:
                sent.append(reminder)
    return sent
