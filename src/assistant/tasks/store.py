"""SQLite-backed store for the assistant's to-do list.

A single ``tasks`` table in its own SQLite file (:attr:`Settings.tasks_db_path`,
under the memory directory), modeled on :mod:`assistant.calendar.store`. A task is
distinct from a calendar event: it has no fixed time (an *optional* ``due``), and
it carries a ``done`` state. ``due`` — when set — is a timezone-aware ISO-8601
string, stored so the offset travels with the value, exactly as the calendar
store does for event times.

A fresh connection is opened per operation with WAL + a busy timeout, so the
store is safe to touch from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime

from ..calendar.store import parse_dt  # shared tz-aware ISO parsing
from ..config import Settings, postgres_backend
from ..sqlite_util import connect, ensure_columns, open_db
from ..timeutil import normalize_stamp as _normalize_due
from ..timeutil import stamp_now as _stamp_now

# Columns a caller may set on create/update (id + timestamps + done_at managed here).
_FIELDS = ("title", "due", "notes", "rrule")

# Columns added after the table's first creation (see _open's cheap migration).
_ADDED_COLUMNS = ("rrule", "notify_only")


def _truthy(value: object) -> bool:
    """Interpret a tool/DB flag value (string, bool, int) as a boolean."""
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Task:
    """A single to-do item.

    ``due`` is an optional tz-aware ISO-8601 string (empty for an undated task).
    ``done`` is the completion state; ``done_at`` is the ISO stamp when it was
    completed (empty while open). ``rrule`` is an optional RFC 5545 recurrence
    rule (e.g. ``FREQ=WEEKLY;BYDAY=SU``) anchored at ``due``: completing a
    recurring task rolls its ``due`` forward to the next occurrence instead of
    closing it (see :func:`complete_task`).
    """

    id: str
    title: str
    done: bool = False
    due: str = ""
    notes: str = ""
    rrule: str = ""
    created: str = ""
    updated: str = ""
    done_at: str = ""
    # A one-time timed reminder that fires at its due time and does NOT keep
    # nagging once overdue (a purely informational "remind me at TIME that X",
    # not a to-do to complete). See tasks.reminders.due_task_reminders.
    notify_only: bool = False


@dataclass
class TaskCompletion:
    """Snapshot of one completed occurrence, retained even after task deletion."""

    id: str
    task_id: str
    title: str
    due: str = ""
    rrule: str = ""
    completed_at: str = ""
    undone_at: str = ""
    occurrence_seq: int = 0


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.tasks_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        " id TEXT PRIMARY KEY, title TEXT NOT NULL, done INTEGER DEFAULT 0,"
        " due TEXT DEFAULT '', notes TEXT DEFAULT '', rrule TEXT DEFAULT '',"
        " created TEXT DEFAULT '', updated TEXT DEFAULT '', done_at TEXT DEFAULT '',"
        " notify_only TEXT DEFAULT '')"
    )
    ensure_columns(conn, "tasks", _ADDED_COLUMNS)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS task_completions ("
        " id TEXT PRIMARY KEY, task_id TEXT NOT NULL, title TEXT NOT NULL,"
        " due TEXT NOT NULL DEFAULT '', rrule TEXT NOT NULL DEFAULT '',"
        " completed_at TEXT NOT NULL, undone_at TEXT NOT NULL DEFAULT '',"
        " occurrence_seq INTEGER NOT NULL DEFAULT 0)"
    )
    completion_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(task_completions)")
    }
    if "occurrence_seq" not in completion_columns:
        conn.execute(
            "ALTER TABLE task_completions"
            " ADD COLUMN occurrence_seq INTEGER NOT NULL DEFAULT 0"
        )
        # Old rows have no durable ordering beyond their insertion order.
        # Assign once, as part of adding the column, preserving rowid for ties.
        conn.execute(
            "UPDATE task_completions AS c SET occurrence_seq ="
            " (SELECT COUNT(*) FROM task_completions x WHERE x.task_id = c.task_id"
            "   AND (x.completed_at < c.completed_at OR"
            "        (x.completed_at = c.completed_at AND x.rowid <= c.rowid)))"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS task_completions_completed_at_idx"
        " ON task_completions(completed_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS task_completions_task_completed_idx"
        " ON task_completions(task_id, completed_at)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO task_completions"
        " (id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq)"
        " SELECT 'legacy:' || id, id, title, due, rrule, done_at, '',"
        " COALESCE((SELECT MAX(c.occurrence_seq) FROM task_completions c"
        " WHERE c.task_id = tasks.id), 0) + 1 FROM tasks"
        " WHERE done = 1 AND done_at <> '' AND NOT EXISTS ("
        " SELECT 1 FROM task_completions c WHERE c.task_id = tasks.id)"
    )
    # _open is also used by read operations; persist idempotent schema/backfill
    # before callers begin their own transaction.
    conn.commit()
    return conn


def _connect(settings: Settings) -> AbstractContextManager[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see sqlite_util.connect)."""
    return connect(_open, settings)


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        title=row["title"],
        done=bool(row["done"]),
        due=row["due"] or "",
        notes=row["notes"] or "",
        rrule=row["rrule"] or "",
        created=row["created"] or "",
        updated=row["updated"] or "",
        done_at=row["done_at"] or "",
        notify_only=_truthy(row["notify_only"] or ""),
    )


def _row_to_completion(row: sqlite3.Row | dict) -> TaskCompletion:
    return TaskCompletion(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        title=str(row["title"]),
        due=str(row["due"] or ""),
        rrule=str(row["rrule"] or ""),
        completed_at=str(row["completed_at"]),
        undone_at=str(row["undone_at"] or ""),
        occurrence_seq=int(row["occurrence_seq"] or 0),
    )


def _sort_key(task: Task) -> tuple[int, float, str]:
    """Open tasks: dated ones first by due instant, then undated (by title)."""
    dt = parse_dt(task.due)
    if dt is None:
        return (1, 0.0, task.title.lower())
    return (0, dt.timestamp(), task.title.lower())


def create_task(
    settings: Settings,
    title: str,
    due: str = "",
    notes: str = "",
    rrule: str = "",
    notify_only: object = False,
) -> Task:
    """Insert a new open task and return it (with a generated id and timestamps)."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.create_task(settings, title, due, notes, rrule, notify_only)
    now = _stamp_now(settings)
    task = Task(
        id=uuid.uuid4().hex[:12],
        title=title.strip(),
        done=False,
        due=_normalize_due(settings, due),
        notes=notes.strip(),
        rrule=rrule.strip(),
        created=now,
        updated=now,
        notify_only=_truthy(notify_only),
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO tasks"
            " (id, title, done, due, notes, rrule, created, updated, done_at, notify_only)"
            " VALUES (?, ?, 0, ?, ?, ?, ?, ?, '', ?)",
            (task.id, task.title, task.due, task.notes, task.rrule,
             task.created, task.updated, "1" if task.notify_only else ""),
        )
    return task


def get_task(settings: Settings, task_id: str) -> Task | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_task(settings, task_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


def list_tasks(settings: Settings, include_done: bool = False) -> list[Task]:
    """Open tasks (soonest due first, undated last). ``include_done`` adds
    completed ones after the open ones."""
    if storage_postgres := postgres_backend(settings):
        tasks = storage_postgres.list_tasks(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM tasks").fetchall()
        tasks = [_row_to_task(r) for r in rows]
    open_tasks = sorted((t for t in tasks if not t.done), key=_sort_key)
    if not include_done:
        return open_tasks
    done_tasks = sorted(
        (t for t in tasks if t.done), key=lambda t: t.done_at, reverse=True
    )
    return open_tasks + done_tasks


def update_task(settings: Settings, task_id: str, **fields: object) -> Task | None:
    """Update the given columns on a task; return it, or ``None`` if absent."""
    # notify_only is a boolean flag, stored as "1"/"" — coerced apart from the
    # plain text _FIELDS so "false" clears it rather than storing the word.
    notify_update: dict[str, str] = {}
    if fields.get("notify_only") is not None:
        notify_update["notify_only"] = "1" if _truthy(fields["notify_only"]) else ""
    if storage_postgres := postgres_backend(settings):
        updates = {k: str(v).strip() for k, v in fields.items() if k in _FIELDS and v is not None}
        updates.update(notify_update)
        return storage_postgres.update_task(settings, task_id, updates)
    updates = {
        k: str(v).strip()
        for k, v in fields.items()
        if k in _FIELDS and v is not None
    }
    if "due" in updates:
        updates["due"] = _normalize_due(settings, updates["due"])
    updates.update(notify_update)
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    if not updates:
        return existing
    updates["updated"] = _stamp_now(settings)
    columns = ", ".join(f"{k} = ?" for k in updates)
    with _connect(settings) as conn:
        conn.execute(
            f"UPDATE tasks SET {columns} WHERE id = ?",
            (*updates.values(), task_id),
        )
    return get_task(settings, task_id)


def next_due(settings: Settings, task: Task) -> str:
    """The recurring task's next due after now (and after its current due), as a
    tz-aware ISO string — ``""`` when the task doesn't recur, its rule is
    exhausted (``UNTIL`` passed), or its rule/due is unusable.

    The rule re-anchors at the current due on every roll, so ``COUNT`` counts
    from the latest completion rather than the task's creation — bound a chore
    with ``UNTIL`` instead.
    """
    from ..calendar.context import now, resolve_tz
    from ..calendar.recurrence import build_rule

    dtstart = parse_dt(task.due)
    if not task.rrule or dtstart is None:
        return ""
    rule = build_rule(task.rrule, dtstart, resolve_tz(settings))
    if rule is None:
        return ""
    upcoming = rule.after(max(now(settings), dtstart))
    return upcoming.isoformat() if upcoming is not None else ""


def complete_task_occurrence(
    settings: Settings, task_id: str, completion_id: str | None = None
) -> tuple[Task | None, bool]:
    """Complete one occurrence and report whether a mutation was applied."""
    completion_id = completion_id or uuid.uuid4().hex
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.complete_task_occurrence(settings, task_id, completion_id)

    conn = _open(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            conn.rollback()
            return None, False
        existing = _row_to_task(row)
        if existing.done:
            conn.rollback()
            return existing, False
        duplicate = conn.execute(
            "SELECT 1 FROM task_completions WHERE id = ?", (completion_id,)
        ).fetchone()
        if duplicate:
            conn.rollback()
            return existing, False

        now = _stamp_now(settings)
        upcoming = next_due(settings, existing)
        occurrence_seq = int(conn.execute(
            "SELECT COALESCE(MAX(occurrence_seq), 0) + 1"
            " FROM task_completions WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0])
        conn.execute(
            "INSERT INTO task_completions"
            " (id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq)"
            " VALUES (?, ?, ?, ?, ?, ?, '', ?)",
            (completion_id, existing.id, existing.title, existing.due, existing.rrule,
             now, occurrence_seq),
        )
        if upcoming:
            conn.execute(
                "UPDATE tasks SET due = ?, updated = ? WHERE id = ?",
                (upcoming, now, task_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET done = 1, done_at = ?, updated = ? WHERE id = ?",
                (now, now, task_id),
            )
        updated_row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        conn.commit()
        return _row_to_task(updated_row), True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def complete_task(
    settings: Settings, task_id: str, completion_id: str | None = None
) -> Task | None:
    """Mark a task done (idempotent); return it, or ``None`` if absent.

    A recurring task (``rrule`` set, next occurrence available) is not closed:
    its ``due`` rolls forward to that occurrence and it stays open — the fired
    ledger keys on ``(task_id, due, lead)``, so reminders re-arm on the new due.
    An exhausted or ruleless task completes normally.
    """
    # Preserve the long-standing backend adapter seam for direct callers and
    # tests; the operation path supplies an id and uses the richer helper.
    if completion_id is None and (storage_postgres := postgres_backend(settings)):
        return storage_postgres.complete_task(settings, task_id)
    task, _ = complete_task_occurrence(settings, task_id, completion_id)
    return task


def list_task_completions(
    settings: Settings,
    *,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    query: str = "",
    include_undone: bool = False,
    limit: int | None = None,
) -> list[TaskCompletion]:
    """List occurrences by actual instant, using the interval ``(since, until]``."""
    if storage_postgres := postgres_backend(settings):
        rows = storage_postgres.list_task_completions(settings)
    else:
        with _connect(settings) as conn:
            db_rows = conn.execute("SELECT * FROM task_completions").fetchall()
        rows = [_row_to_completion(row) for row in db_rows]

    def boundary(value: datetime | str | None) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone() if value.tzinfo is None else value
        return parse_dt(value or "")

    after, through = boundary(since), boundary(until)
    needle = query.strip().lower()
    filtered: list[tuple[float, TaskCompletion]] = []
    for completion in rows:
        when = parse_dt(completion.completed_at)
        if when is None or (completion.undone_at and not include_undone):
            continue
        if after is not None and when <= after:
            continue
        if through is not None and when > through:
            continue
        if needle and completion.task_id != query.strip() and needle not in completion.title.lower():
            continue
        filtered.append((when.timestamp(), completion))
    filtered.sort(
        key=lambda item: (item[0], item[1].occurrence_seq, item[1].id),
        reverse=True,
    )
    completions = [item[1] for item in filtered]
    if limit is not None:
        completions = completions[: max(0, min(int(limit), 1000))]
    return completions


def restore_task_completion(settings: Settings, completion: TaskCompletion) -> TaskCompletion:
    """Upsert a completion snapshot verbatim (used by local-to-Postgres import)."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.restore_task_completion(settings, completion)
    with _connect(settings) as conn:
        legacy_id = f"legacy:{completion.task_id}"
        if completion.id != legacy_id:
            conn.execute(
                "DELETE FROM task_completions WHERE id = ?", (legacy_id,)
            )
        occurrence_seq = completion.occurrence_seq
        if occurrence_seq <= 0:
            existing = conn.execute(
                "SELECT occurrence_seq FROM task_completions WHERE id = ?",
                (completion.id,),
            ).fetchone()
            occurrence_seq = int(existing[0]) if existing else int(conn.execute(
                "SELECT COALESCE(MAX(occurrence_seq), 0) + 1"
                " FROM task_completions WHERE task_id = ?",
                (completion.task_id,),
            ).fetchone()[0])
            completion.occurrence_seq = occurrence_seq
        conn.execute(
            "INSERT OR REPLACE INTO task_completions"
            " (id, task_id, title, due, rrule, completed_at, undone_at, occurrence_seq)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (completion.id, completion.task_id, completion.title, completion.due,
             completion.rrule, completion.completed_at, completion.undone_at,
             occurrence_seq),
        )
    return completion


def restore_task(settings: Settings, task: Task) -> Task:
    """Re-insert a full task snapshot verbatim, overwriting any current row with
    the same id. Used by the undo path (see :mod:`.undo`); never bumps ``updated``."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.restore_task(settings, task)
    with _connect(settings) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tasks"
            " (id, title, done, due, notes, rrule, created, updated, done_at, notify_only)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task.id, task.title, int(task.done), task.due, task.notes,
                task.rrule, task.created, task.updated, task.done_at,
                "1" if task.notify_only else "",
            ),
        )
    return task


def restore_task_and_undo_completion(
    settings: Settings, task: Task, completion_id: str
) -> Task:
    """Atomically restore ``task`` and mark its exact completion occurrence undone."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.restore_task_and_undo_completion(
            settings, task, completion_id
        )
    with _connect(settings) as conn:
        conn.execute("BEGIN IMMEDIATE")
        completion_rows = conn.execute(
            "SELECT * FROM task_completions WHERE task_id = ?",
            (task.id,),
        ).fetchall()
        target = next(
            (
                row for row in completion_rows
                if row["id"] == completion_id and not row["undone_at"]
            ),
            None,
        )
        if target is None:
            raise ValueError(f"completion not found or already undone: {completion_id}")
        target_seq = int(target["occurrence_seq"])
        active_rows = [
            row for row in completion_rows
            if row["id"] != completion_id and not row["undone_at"]
        ]
        has_later = any(
            int(row["occurrence_seq"]) > target_seq for row in active_rows
        )
        if not has_later:
            # The write-ledger snapshot belongs to this completion's original
            # pre-state.  If later occurrences were undone first, that due can
            # still include advances from occurrences which are no longer
            # active.  The first historical snapshot after the last remaining
            # active occurrence is the exact due at that boundary.
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
                "INSERT OR REPLACE INTO tasks"
                " (id, title, done, due, notes, rrule, created, updated, done_at, notify_only)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (task.id, task.title, int(task.done), task.due, task.notes, task.rrule,
                 task.created, task.updated, task.done_at,
                 "1" if task.notify_only else ""),
            )
        cursor = conn.execute(
            "UPDATE task_completions SET undone_at = ? WHERE id = ? AND undone_at = ''",
            (_stamp_now(settings), completion_id),
        )
        if cursor.rowcount != 1:  # defensive: target was locked by this write txn
            raise ValueError(f"completion could not be undone: {completion_id}")
    return task


def delete_task(settings: Settings, task_id: str) -> Task | None:
    """Delete a task by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_task(settings, task_id)
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    return existing


def find_tasks(settings: Settings, query: str) -> list[Task]:
    """All candidate tasks for ``query``: an exact-id match alone, else every
    case-insensitive title-substring match. Open tasks shadow completed ones
    (completed tasks are only returned when nothing open matches), mirroring the
    calendar's upcoming-shadows-past rule."""
    query = query.strip()
    if not query:
        return []
    exact = get_task(settings, query)
    if exact is not None:
        return [exact]
    needle = query.lower()
    matches = [t for t in list_tasks(settings, include_done=True) if needle in t.title.lower()]
    if not matches:
        return []
    open_matches = [t for t in matches if not t.done]
    return open_matches or matches


def find_task(settings: Settings, query: str) -> Task | None:
    """Resolve ``query`` to a single task: by exact id, else the best title match."""
    matches = find_tasks(settings, query)
    return matches[0] if matches else None


def find_exact_open_title(settings: Settings, title: str) -> Task | None:
    """The open task whose title exactly matches ``title`` (case-insensitive,
    stripped), or None. Unlike find_tasks's substring fuzz, this is strict —
    used solely to dedupe add_task ("Buy milk" must not collide with "Buy
    milk and eggs"). Backend-dispatches via list_tasks like its neighbors."""
    needle = title.strip().lower()
    if not needle:
        return None
    for t in list_tasks(settings, include_done=False):
        if t.title.strip().lower() == needle:
            return t
    return None
