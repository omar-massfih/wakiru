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
              rrule TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT '',
              done_at TEXT NOT NULL DEFAULT '',
              notify_only TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # Tables created before a column existed need it added in place.
        conn.execute(
            "ALTER TABLE assistant_tasks"
            " ADD COLUMN IF NOT EXISTS rrule TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(
            "ALTER TABLE assistant_tasks"
            " ADD COLUMN IF NOT EXISTS notify_only TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_task_completions (
              id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              title TEXT NOT NULL,
              due TEXT NOT NULL DEFAULT '',
              rrule TEXT NOT NULL DEFAULT '',
              completed_at TEXT NOT NULL,
              undone_at TEXT NOT NULL DEFAULT '',
              occurrence_seq BIGINT NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "ALTER TABLE assistant_task_completions"
            " ADD COLUMN IF NOT EXISTS occurrence_seq BIGINT NOT NULL DEFAULT 0"
        )
        conn.execute(
            """
            WITH unsequenced AS (
              SELECT id, task_id, completed_at
              FROM assistant_task_completions WHERE occurrence_seq = 0
            ), ranked AS (
              SELECT u.id, ROW_NUMBER() OVER (
                PARTITION BY u.task_id ORDER BY u.completed_at, u.id
              ) + COALESCE((
                SELECT MAX(c.occurrence_seq)
                FROM assistant_task_completions c
                WHERE c.task_id = u.task_id AND c.occurrence_seq > 0
              ), 0) AS seq
              FROM unsequenced u
            )
            UPDATE assistant_task_completions c SET occurrence_seq = ranked.seq
            FROM ranked WHERE c.id = ranked.id
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS assistant_task_completions_completed_at_idx "
            "ON assistant_task_completions(completed_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS assistant_task_completions_task_completed_idx "
            "ON assistant_task_completions(task_id, completed_at)"
        )
        conn.execute(
            """
            INSERT INTO assistant_task_completions
              (id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq)
            SELECT 'legacy:' || t.id, t.id, t.title, t.due, t.rrule, t.done_at, '',
              COALESCE((SELECT MAX(c.occurrence_seq)
                        FROM assistant_task_completions c
                        WHERE c.task_id = t.id), 0) + 1
            FROM assistant_tasks t WHERE done = TRUE AND done_at <> ''
              AND NOT EXISTS (
                SELECT 1 FROM assistant_task_completions c WHERE c.task_id = t.id
              )
            ON CONFLICT(id) DO NOTHING
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
        rrule=str(row.get("rrule") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
        done_at=str(row.get("done_at") or ""),
        notify_only=str(row.get("notify_only") or "").strip().lower()
        in ("1", "true", "yes", "on"),
    )


def _completion_from_row(row: dict):
    from ..tasks.store import TaskCompletion

    return TaskCompletion(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        title=str(row["title"]),
        due=str(row.get("due") or ""),
        rrule=str(row.get("rrule") or ""),
        completed_at=str(row["completed_at"]),
        undone_at=str(row.get("undone_at") or ""),
        occurrence_seq=int(row.get("occurrence_seq") or 0),
    )


def create_task(
    settings: Settings,
    title: str,
    due: str = "",
    notes: str = "",
    rrule: str = "",
    notify_only: object = False,
):
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
        rrule=rrule.strip(),
        created=now,
        updated=now,
        notify_only=task_store._truthy(notify_only),
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_tasks (id, title, done, due, notes, rrule, created, updated, done_at, notify_only) "
            "VALUES (%s, %s, FALSE, %s, %s, %s, %s, %s, '', %s)",
            (task.id, task.title, task.due, task.notes, task.rrule, task.created,
             task.updated, "1" if task.notify_only else ""),
        )
    return task


def get_task(settings: Settings, task_id: str):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, done, due, notes, rrule, created, updated, done_at, notify_only FROM assistant_tasks WHERE id = %s", (task_id,)))
    return _task_from_row(rows[0]) if rows else None


def list_tasks(settings: Settings):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute("SELECT id, title, done, due, notes, rrule, created, updated, done_at, notify_only FROM assistant_tasks"))
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


def complete_task_occurrence(settings: Settings, task_id: str, completion_id: str):
    from ..tasks import store as task_store

    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(
            "SELECT id, title, done, due, notes, rrule, created, updated, done_at, "
            "notify_only FROM assistant_tasks WHERE id = %s FOR UPDATE", (task_id,)
        ))
        if not rows:
            return None, False
        existing = _task_from_row(rows[0])
        if existing.done:
            return existing, False
        duplicate = _rows(conn.execute(
            "SELECT id FROM assistant_task_completions WHERE id = %s", (completion_id,)
        ))
        if duplicate:
            return existing, False
        now = task_store._stamp_now(settings)
        upcoming = task_store.next_due(settings, existing)
        sequence_rows = _rows(conn.execute(
            "SELECT COALESCE(MAX(occurrence_seq), 0) + 1 AS occurrence_seq "
            "FROM assistant_task_completions WHERE task_id = %s",
            (task_id,),
        ))
        occurrence_seq = int(sequence_rows[0]["occurrence_seq"])
        conn.execute(
            "INSERT INTO assistant_task_completions "
            "(id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq) "
            "VALUES (%s, %s, %s, %s, %s, %s, '', %s)",
            (completion_id, existing.id, existing.title, existing.due,
             existing.rrule, now, occurrence_seq),
        )
        if upcoming:
            conn.execute("UPDATE assistant_tasks SET due = %s, updated = %s WHERE id = %s", (upcoming, now, task_id))
            existing.due = upcoming
            existing.updated = now
        else:
            conn.execute("UPDATE assistant_tasks SET done = TRUE, done_at = %s, updated = %s WHERE id = %s", (now, now, task_id))
            existing.done = True
            existing.done_at = now
            existing.updated = now
    return existing, True


def complete_task(settings: Settings, task_id: str, completion_id: str | None = None):
    import uuid

    task, _ = complete_task_occurrence(
        settings, task_id, completion_id or uuid.uuid4().hex
    )
    return task


def list_task_completions(settings: Settings):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(
            "SELECT id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq "
            "FROM assistant_task_completions"
        ))
    return [_completion_from_row(row) for row in rows]


def restore_task_completion(settings: Settings, completion):
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        legacy_id = f"legacy:{completion.task_id}"
        if completion.id != legacy_id:
            conn.execute(
                "DELETE FROM assistant_task_completions WHERE id = %s",
                (legacy_id,),
            )
        occurrence_seq = completion.occurrence_seq
        if occurrence_seq <= 0:
            rows = _rows(conn.execute(
                "SELECT occurrence_seq FROM assistant_task_completions "
                "WHERE id = %s",
                (completion.id,),
            ))
            if not rows:
                rows = _rows(conn.execute(
                    "SELECT COALESCE(MAX(occurrence_seq), 0) + 1 AS occurrence_seq "
                    "FROM assistant_task_completions WHERE task_id = %s",
                    (completion.task_id,),
                ))
            occurrence_seq = int(rows[0]["occurrence_seq"])
            completion.occurrence_seq = occurrence_seq
        conn.execute(
            """
            INSERT INTO assistant_task_completions
              (id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id) DO UPDATE SET
              task_id = excluded.task_id, title = excluded.title,
              due = excluded.due, rrule = excluded.rrule,
              completed_at = excluded.completed_at, undone_at = excluded.undone_at,
              occurrence_seq = excluded.occurrence_seq
            """,
            (completion.id, completion.task_id, completion.title, completion.due,
             completion.rrule, completion.completed_at, completion.undone_at,
             occurrence_seq),
        )
    return completion


def restore_task(settings: Settings, task) -> object:
    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_tasks (id, title, done, due, notes, rrule, created, updated, done_at, notify_only)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id) DO UPDATE SET
              title = excluded.title,
              done = excluded.done,
              due = excluded.due,
              notes = excluded.notes,
              rrule = excluded.rrule,
              created = excluded.created,
              updated = excluded.updated,
              done_at = excluded.done_at,
              notify_only = excluded.notify_only
            """,
            (task.id, task.title, bool(task.done), task.due, task.notes, task.rrule,
             task.created, task.updated, task.done_at, "1" if task.notify_only else ""),
        )
    return task


def restore_task_and_undo_completion(settings: Settings, task, completion_id: str):
    from ..tasks import store as task_store

    ensure_tasks_schema(settings)
    with connect(settings) as conn:
        # Completion takes the task lock before touching its history.  Undo must
        # use the same lock order so it cannot inspect stale history and then
        # overwrite a task advanced by a concurrent completion.
        conn.execute(
            "SELECT id FROM assistant_tasks WHERE id = %s FOR UPDATE",
            (task.id,),
        )
        completion_rows = _rows(conn.execute(
            "SELECT id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq "
            "FROM assistant_task_completions WHERE task_id = %s "
            "FOR UPDATE",
            (task.id,),
        ))
        target = next(
            (
                row for row in completion_rows
                if str(row["id"]) == completion_id and not row["undone_at"]
            ),
            None,
        )
        if target is None:
            raise ValueError(f"completion not found or already undone: {completion_id}")
        target_seq = int(target["occurrence_seq"])
        active_rows = [
            row for row in completion_rows
            if str(row["id"]) != completion_id and not row["undone_at"]
        ]
        has_later = any(
            int(row["occurrence_seq"]) > target_seq
            for row in active_rows
        )
        if not has_later:
            # A later completion's write-ledger snapshot may include due-date
            # advances from occurrences that have since been undone.  Rebase
            # the due on the durable occurrence sequence, including undone
            # rows as boundary snapshots, so out-of-order undos compose.
            last_active_seq = max(
                (int(row["occurrence_seq"]) for row in active_rows), default=0
            )
            boundary = min(
                (
                    row for row in completion_rows
                    if int(row["occurrence_seq"]) > last_active_seq
                ),
                key=lambda row: int(row["occurrence_seq"]),
            )
            task.due = str(boundary["due"] or "")
            conn.execute(
                """
                INSERT INTO assistant_tasks (id, title, done, due, notes, rrule, created, updated, done_at, notify_only)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                  title = excluded.title, done = excluded.done, due = excluded.due,
                  notes = excluded.notes, rrule = excluded.rrule,
                  created = excluded.created, updated = excluded.updated,
                  done_at = excluded.done_at, notify_only = excluded.notify_only
                """,
                (task.id, task.title, bool(task.done), task.due, task.notes, task.rrule,
                 task.created, task.updated, task.done_at,
                 "1" if task.notify_only else ""),
            )
        cursor = conn.execute(
            "UPDATE assistant_task_completions SET undone_at = %s "
            "WHERE id = %s AND undone_at = ''",
            (task_store._stamp_now(settings), completion_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"completion not found or already undone: {completion_id}")
    return task


def delete_task(settings: Settings, task_id: str):
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_tasks WHERE id = %s", (task_id,))
    return existing
