"""Task writes — the operation set the agent's task tools apply.

Each operation arrives as a parsed dict —

* ``add``      — create a new task,
* ``complete`` — mark an existing task done (a recurring one rolls forward),
* ``update``   — change an existing task's title/due/notes/rrule,
* ``remove``   — delete a task.

Existing tasks are targeted by id (with a fuzzy title fallback that refuses
ambiguity, exactly as :mod:`assistant.calendar.ops` does). Every applied op is
logged to the undo ledger under the turn's batch.
"""

from __future__ import annotations

from .. import write_ops
from ..calendar import recurrence
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


_RULE_NOTE = " (recurrence ignored: needs a valid RRULE and a parseable due date)"


def _usable_rrule(settings: Settings, rule: str, due: str) -> bool:
    """A task recurrence needs a parseable rule and a parseable due to anchor —
    an unparseable due would leave a rule that can never roll forward."""
    return (
        recurrence.validate_rrule(rule)
        and store.parse_dt(store._normalize_due(settings, due)) is not None
    )


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
        rrule = str(op.get("rrule", "") or "")
        note = ""
        if rrule and not _usable_rrule(settings, rrule, str(op.get("due", "") or "")):
            rrule, note = "", _RULE_NOTE  # keep the task, tell the model why
        task = store.create_task(
            settings,
            title=str(op["title"]),
            due=str(op.get("due", "") or ""),
            notes=str(op.get("notes", "") or ""),
            rrule=rrule,
        )
        suffix = f" ({recurrence.humanize_rrule(task.rrule)})" if task.rrule else ""
        summary = f"added task: {task.title}{suffix}{note}"
        _log_write(settings, thread_id, batch_id, task.id, "add", summary, None)
        return summary

    if kind == "complete":
        target = _target_id(settings, op)
        if target is None:
            return None
        before = store.get_task(settings, target)
        if (
            before is not None and before.rrule
            and undo.completed_in_batch(settings, thread_id, batch_id, target)
        ):
            # A doubled complete in one turn would roll the due forward twice,
            # silently skipping an occurrence — treat it as the duplicate it is.
            return f"already completed this turn: {before.title}"
        done = store.complete_task(settings, target)
        if done is None:
            return None
        if not done.done:
            # Recurring task rolled forward instead of closing.
            from ..calendar.context import format_when

            summary = f"completed: {done.title} (recurs — next due {format_when(settings, done.due)})"
        else:
            summary = f"completed: {done.title}"
        _log_write(settings, thread_id, batch_id, target, "complete", summary, before)
        return summary

    if kind == "update":
        target = _target_id(settings, op)
        if target is None:
            return None
        before = store.get_task(settings, target)
        new_rule = op.get("rrule")
        note = ""
        if new_rule is not None and str(new_rule).strip():
            due = str(op.get("due") or (before.due if before else ""))
            if not _usable_rrule(settings, str(new_rule), due):
                new_rule, note = None, _RULE_NOTE  # leave any existing rule untouched
        revised = store.update_task(
            settings, target,
            title=op.get("title"), due=op.get("due"), notes=op.get("notes"),
            rrule=new_rule,
        )
        if revised is None:
            return None
        summary = f"updated task: {revised.title}{note}"
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
