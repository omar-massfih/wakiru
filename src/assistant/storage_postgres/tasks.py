"""Task tables for the Postgres backend."""

from __future__ import annotations

from ..config import Settings
from .core import (
    _rows,
    _schema_done,
    _schema_mark,
    connect,
)


def ensure_tasks_schema(settings: Settings) -> None:
    if _schema_done(settings, "tasks"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_tasks (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              done BOOLEAN NOT NULL DEFAULT FALSE,
              due TEXT NOT NULL DEFAULT '',
              notes TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT '',
              done_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_task_write_log (
              id BIGSERIAL PRIMARY KEY,
              thread_id TEXT NOT NULL,
              batch_id TEXT NOT NULL,
              task_id TEXT NOT NULL,
              op TEXT NOT NULL,
              summary TEXT NOT NULL,
              before_json TEXT,
              applied_at TEXT NOT NULL,
              undone_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_task_reminders_fired (
              task_id TEXT NOT NULL,
              due TEXT NOT NULL,
              lead_minutes INTEGER NOT NULL,
              fired_at TEXT NOT NULL,
              PRIMARY KEY (task_id, due, lead_minutes)
            )
            """
        )
    _schema_mark(settings, "tasks")


def _task_from_row(row: dict):
    from ..tasks.store import Task

    return Task(
        id=str(row["id"]),
        title=str(row["title"]),
        done=bool(row.get("done")),
        due=str(row.get("due") or ""),
        notes=str(row.get("notes") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
        done_at=str(row.get("done_at") or ""),
    )


def create_task(settings: Settings, title: str, due: str = "", notes: str = ""):
    import uuid

    from ..tasks import store as task_store

    ensure_tasks_schema(settings)
    now = task_store._stamp_now(settings)
    task = task_store.Task(
        id=uuid.uuid4().hex[:12],
        title=title.strip(),
        done=False,
        due=task_store._normalize_due(settings, due),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_tasks (id, title, done, due, notes, created, updated, done_at) "
            "VALUES (%s, %s, FALSE, %s, %s, %s, %s, '')",
            (task.id, task.title, task.due, task.notes, task.created, task.updated),
        )
    return task


def get_task(settings: Settings, task_id: str):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, done, due, notes, created, updated, done_at FROM assistant_tasks WHERE id = %s", (task_id,)))
    return _task_from_row(rows[0]) if rows else None


def list_tasks(settings: Settings):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, done, due, notes, created, updated, done_at FROM assistant_tasks"))
    return [_task_from_row(r) for r in rows]


def update_task(settings: Settings, task_id: str, fields: dict[str, str]):
    from ..tasks import store as task_store

    ensure_tasks_schema(settings)
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    updates = {k: str(v).strip() for k, v in fields.items() if v is not None}
    if "due" in updates:
        updates["due"] = task_store._normalize_due(settings, updates["due"])
    if not updates:
        return existing
    updates["updated"] = task_store._stamp_now(settings)
    assignments = ", ".join(f"{k} = %s" for k in updates)
    with connect(settings) as conn:
        conn.execute(f"UPDATE assistant_tasks SET {assignments} WHERE id = %s", (*updates.values(), task_id))
    return get_task(settings, task_id)


def complete_task(settings: Settings, task_id: str):
    from ..tasks import store as task_store

    existing = get_task(settings, task_id)
    if existing is None or existing.done:
        return existing
    now = task_store._stamp_now(settings)
    with connect(settings) as conn:
        conn.execute("UPDATE assistant_tasks SET done = TRUE, done_at = %s, updated = %s WHERE id = %s", (now, now, task_id))
    return get_task(settings, task_id)


def restore_task(settings: Settings, task) -> object:
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_tasks (id, title, done, due, notes, created, updated, done_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id) DO UPDATE SET
              title = excluded.title,
              done = excluded.done,
              due = excluded.due,
              notes = excluded.notes,
              created = excluded.created,
              updated = excluded.updated,
              done_at = excluded.done_at
            """,
            (task.id, task.title, bool(task.done), task.due, task.notes, task.created, task.updated, task.done_at),
        )
    return task


def delete_task(settings: Settings, task_id: str):
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_tasks WHERE id = %s", (task_id,))
    return existing
