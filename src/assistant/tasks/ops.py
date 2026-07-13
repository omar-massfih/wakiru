"""Task writes — the operation set the agent's task tools apply.

Each operation arrives as a parsed dict —

* ``add``      — create a new task,
* ``complete`` — mark an existing task done,
* ``update``   — change an existing task's title/due/notes,
* ``remove``   — delete a task.

Existing tasks are targeted by id (with a fuzzy title fallback that refuses
ambiguity, exactly as :mod:`assistant.calendar.ops` does). Every applied op is
logged to the undo ledger under the turn's batch.
"""

from __future__ import annotations

import logging

from ..config import Settings
from . import store, undo

logger = logging.getLogger(__name__)


# Ops whose schema defines "title" as the NEW value, not an identifier. Looking
# the target up by it would resolve to whichever task already bears the new name
# — a row the user never referred to.
_TITLE_IS_NEW_VALUE = {"update"}


def _target_id(settings: Settings, op: dict) -> str | None:
    """Resolve the task an op refers to, by id or a fuzzy title fallback.

    A fuzzy reference matching more than one task is skipped rather than guessed
    at — the same ambiguity guard the calendar extractor uses."""
    ident = op.get("id") or op.get("query")
    if not ident and op["op"] not in _TITLE_IS_NEW_VALUE:
        ident = op.get("title")
    if not ident:
        return None
    matches = store.find_tasks(settings, str(ident))
    if len(matches) > 1:
        logger.warning(
            "task %s target %r is ambiguous between %d tasks (%s); skipping",
            op.get("op"), ident, len(matches),
            ", ".join(t.title for t in matches[:5]),
        )
        return None
    return matches[0].id if matches else None


def _log_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    task_id: str,
    op: str,
    summary: str,
    before: store.Task | None,
) -> None:
    if not (thread_id and batch_id and settings.enable_write_confirmation):
        return
    undo.record_write(settings, thread_id, batch_id, task_id, op, summary, before)


def apply_op(
    settings: Settings, op: dict, thread_id: str = "", batch_id: str = ""
) -> str | None:
    """Apply a single parsed operation; return a short log line, or ``None``."""
    kind = op["op"]
    if kind == "add" and op.get("title"):
        task = store.create_task(
            settings,
            title=str(op["title"]),
            due=str(op.get("due", "") or ""),
            notes=str(op.get("notes", "") or ""),
        )
        summary = f"added task: {task.title}"
        _log_write(settings, thread_id, batch_id, task.id, "add", summary, None)
        return summary

    if kind == "complete":
        target = _target_id(settings, op)
        if target is None:
            return None
        before = store.get_task(settings, target)
        done = store.complete_task(settings, target)
        if done is None:
            return None
        summary = f"completed: {done.title}"
        _log_write(settings, thread_id, batch_id, target, "complete", summary, before)
        return summary

    if kind == "update":
        target = _target_id(settings, op)
        if target is None:
            return None
        before = store.get_task(settings, target)
        revised = store.update_task(
            settings, target,
            title=op.get("title"), due=op.get("due"), notes=op.get("notes"),
        )
        if revised is None:
            return None
        summary = f"updated task: {revised.title}"
        _log_write(settings, thread_id, batch_id, target, "update", summary, before)
        return summary

    if kind == "remove":
        target = _target_id(settings, op)
        if target is None:
            return None
        deleted = store.delete_task(settings, target)
        if deleted is None:
            return None
        summary = f"removed task: {deleted.title}"
        _log_write(settings, thread_id, batch_id, target, "remove", summary, deleted)
        return summary

    return None
