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

from .. import write_ops
from ..config import Settings
from . import store, undo

# The find/record_write lambdas resolve store/undo attributes at call time so
# test monkeypatches on those modules keep working (the same rationale as
# write_ledger.LedgerSpec naming its pg adapters by string).
_SPEC: write_ops.WriteOpsSpec[store.Task] = write_ops.WriteOpsSpec(
    kind="task",
    noun="tasks",
    find=lambda settings, ident: store.find_tasks(settings, ident),
    title_is_new_value=frozenset({"update"}),  # "update" carries the new title
    record_write=lambda *args: undo.record_write(*args),
)


def _target_id(settings: Settings, op: dict) -> str | None:
    return write_ops.resolve_target(_SPEC, settings, op)


def _log_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    task_id: str,
    op: str,
    summary: str,
    before: store.Task | None,
) -> None:
    write_ops.log_write(_SPEC, settings, thread_id, batch_id, task_id, op, summary, before)


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
