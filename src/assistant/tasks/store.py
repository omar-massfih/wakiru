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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from ..calendar.store import parse_dt  # shared tz-aware ISO parsing
from ..config import Settings, postgres_backend
from ..sqlite_util import ensure_columns, open_db, transaction

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
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see calendar.store)."""
    with transaction(_open(settings)) as conn:
        yield conn


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


def _normalize_due(settings: Settings, value: str) -> str:
    """Attach the assistant's timezone to a naive ISO due date on its way in.

    Blank or unparseable values pass through unchanged (filtered on read).
    Mirrors calendar.store._normalize_stamp.
    """
    value = value.strip()
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is not None:
        return value
    from ..calendar.context import resolve_tz

    return dt.replace(tzinfo=resolve_tz(settings)).isoformat()


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


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


def complete_task(settings: Settings, task_id: str) -> Task | None:
    """Mark a task done (idempotent); return it, or ``None`` if absent.

    A recurring task (``rrule`` set, next occurrence available) is not closed:
    its ``due`` rolls forward to that occurrence and it stays open — the fired
    ledger keys on ``(task_id, due, lead)``, so reminders re-arm on the new due.
    An exhausted or ruleless task completes normally.
    """
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.complete_task(settings, task_id)
    existing = get_task(settings, task_id)
    if existing is None:
        return None
    if existing.done:
        return existing
    now = _stamp_now(settings)
    upcoming = next_due(settings, existing)
    with _connect(settings) as conn:
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
    return get_task(settings, task_id)


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
