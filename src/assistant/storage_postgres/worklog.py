"""Work-log table for the Postgres backend (twin of assistant.worklog.store).

Unlike the habit/expense twins this one takes the already-built entry: the
store owns the timer semantics (auto-stop on start, duration on stop) for both
backends, and only the row operations differ.
"""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect

_COLS = "id, project, minutes, note, worked_on, started, ended, created"


def ensure_worklog_schema(settings: Settings) -> None:
    if _schema_done(settings, "worklog"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_work_log (
              id TEXT PRIMARY KEY,
              project TEXT NOT NULL,
              minutes INTEGER NOT NULL DEFAULT 0,
              note TEXT NOT NULL DEFAULT '',
              worked_on TEXT NOT NULL DEFAULT '',
              started TEXT NOT NULL DEFAULT '',
              ended TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "worklog")


def _entry_from_row(row: dict):
    from ..worklog.store import WorkEntry

    return WorkEntry(
        id=str(row["id"]),
        project=str(row["project"]),
        minutes=int(row.get("minutes") or 0),
        note=str(row.get("note") or ""),
        worked_on=str(row.get("worked_on") or ""),
        started=str(row.get("started") or ""),
        ended=str(row.get("ended") or ""),
        created=str(row.get("created") or ""),
    )


def insert_work_entry(settings: Settings, entry):
    ensure_worklog_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_work_log"
            " (id, project, minutes, note, worked_on, started, ended, created)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (entry.id, entry.project, entry.minutes, entry.note, entry.worked_on,
             entry.started, entry.ended, entry.created),
        )
    return entry


def finish_work_entry(
    settings: Settings, entry_id: str, ended: str, minutes: int, note: str
) -> None:
    ensure_worklog_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "UPDATE assistant_work_log SET ended = %s, minutes = %s, note = %s"
            " WHERE id = %s",
            (ended, minutes, note, entry_id),
        )


def get_work_entry(settings: Settings, entry_id: str):
    ensure_worklog_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(f"SELECT {_COLS} FROM assistant_work_log WHERE id = %s", (entry_id,))
        )
    return _entry_from_row(rows[0]) if rows else None


def list_work_entries(settings: Settings):
    ensure_worklog_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_work_log"))
    return [_entry_from_row(r) for r in rows]


def delete_work_entry(settings: Settings, entry_id: str):
    existing = get_work_entry(settings, entry_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_work_log WHERE id = %s", (entry_id,))
    return existing
