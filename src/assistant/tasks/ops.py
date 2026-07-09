"""Task formation — the write path, modeled on :mod:`assistant.calendar.ops`.

After each exchange (in the background, off the reply path) a reconciling
extractor reads the turn together with the current time and the open to-do list,
and returns operations —

* ``add``      — create a new task,
* ``complete`` — mark an existing task done,
* ``update``   — change an existing task's title/due/notes,
* ``remove``   — delete a task.

Existing tasks are targeted by id (with a fuzzy title fallback, ambiguity-guarded
exactly as the calendar extractor does). Best-effort: any failure is logged and
swallowed so the chat reply is never affected.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from .. import notify
from ..codex_runner import run_codex
from ..config import Settings, get_settings
from . import store, undo
from .context import render_tasks
from ..calendar.context import now

logger = logging.getLogger(__name__)


_TASKS_PROMPT = """\
You maintain the to-do list of a personal assistant. Read the exchange, the
current time, and the open tasks, then decide what should change.

Only act on a clear task intent — the user asking to remember/track something to
do, marking something done, or changing/removing a task (or the assistant
confirming it). A task has no fixed meeting time; if the user is scheduling an
event at a specific time, that belongs to the calendar, not here. Ignore
chit-chat and questions that merely ask what's on the list.

A task may have an optional due date. Resolve relative dates ("by Friday",
"tomorrow") against the CURRENT TIME below and emit an absolute ISO-8601 datetime
with offset. To complete, update, or remove an existing task, reference it by its
exact id from the list below.

Return a JSON array of operations, each one of:
  {{"op": "add", "title": "<short title>", "due": "<ISO-8601 with offset, or omit>", "notes": "<or omit>"}}
  {{"op": "complete", "id": "<existing task id>"}}
  {{"op": "update", "id": "<existing task id>", "title": "<or omit>", "due": "<ISO-8601, or omit>", "notes": "<or omit>"}}
  {{"op": "remove", "id": "<existing task id>"}}
Return [] if nothing should change. Output JSON only — no prose, no code fences.

CURRENT TIME: {now}

Open tasks:
{tasks}

User: {user}
Assistant: {assistant}
"""


def _parse_ops(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [
        d for d in data
        if isinstance(d, dict) and d.get("op") in {"add", "complete", "update", "remove"}
    ]


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


def update_tasks(
    settings: Settings | None, user_msg: str, assistant_msg: str, thread_id: str = ""
) -> list[str]:
    """Extract and apply task operations for one turn (add/complete/update/remove).

    Intended to run in the background — it makes a second Codex call. Returns a
    short log of what changed. No-ops when ``enable_auto_tasks`` is false. When
    ``thread_id`` is given and ``enable_write_confirmation`` is on, every applied
    op is logged to the undo ledger under one batch and an out-of-band
    confirmation (with an undo hint) is pushed back to that thread.
    """
    settings = settings or get_settings()
    if not settings.enable_auto_tasks:
        return []

    prompt = _TASKS_PROMPT.format(
        now=now(settings).isoformat(timespec="minutes"),
        tasks=render_tasks(settings, store.list_tasks(settings), with_ids=True),
        user=user_msg,
        assistant=assistant_msg,
    )
    try:
        raw = run_codex(prompt, settings=settings)
    except Exception:
        logger.exception("task extraction (run_codex) failed; skipping this turn")
        return []

    batch_id = uuid.uuid4().hex if thread_id else ""
    applied: list[str] = []
    for op in _parse_ops(raw):
        try:
            result = apply_op(settings, op, thread_id, batch_id)
            if result:
                applied.append(result)
        except Exception:
            logger.exception("failed to apply task op: %s", op)

    if applied:
        logger.info("tasks updated: %s", "; ".join(applied))
        if thread_id and settings.enable_write_confirmation:
            try:
                message = "\n".join(applied) + (
                    f'\nReply "undo" within {settings.write_undo_window_minutes} min to revert.'
                )
                notify.deliver_write_confirmation(settings, thread_id, message)
            except Exception:
                logger.exception("failed to push task confirmation for thread %s", thread_id)
    return applied
