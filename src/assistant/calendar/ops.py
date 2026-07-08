"""Calendar formation — the write path, modeled on :mod:`assistant.memory.learn`.

After each exchange (in the background, off the reply path) a reconciling
extractor reads the turn *together with the current time and what's already
booked* and returns operations —

* ``create``     — schedule a new event,
* ``reschedule`` — change an existing event's time/details (an in-place update),
* ``cancel``     — remove an event.

Because the extractor is given the real current time and the upcoming events (with
their ids), it resolves natural-language dates ("this Friday at 3pm") to concrete
ISO-8601 datetimes and targets existing events by id, so it neither double-books
nor invents times. Like memory upkeep it is best-effort: any failure is logged and
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
from . import recurrence, store, undo
from .context import now, render_events, resolve_tz, writer_view

logger = logging.getLogger(__name__)


_SCHEDULE_PROMPT = """\
You maintain the local calendar of a personal assistant. Read the exchange, the
current time, and the events already scheduled, then decide what should change.

Only act on a clear scheduling intent — the user asking to book, move, or cancel
something (or the assistant confirming it). Ignore chit-chat and questions that
merely ask about the schedule. Resolve relative dates ("tomorrow", "this Friday
at 3pm") against the CURRENT TIME below and always emit absolute ISO-8601
datetimes that include the timezone offset. To move or cancel an existing event,
reference it by its exact id from the list below.

For a repeating event ("every Monday", "daily standup", "weekly 1:1"), set "rrule"
to an RFC 5545 recurrence rule and use "start" as the first occurrence — e.g. every
Monday → "FREQ=WEEKLY;BYDAY=MO", every weekday → "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
daily → "FREQ=DAILY". A series shows its rule in the list below; reschedule and
cancel act on the WHOLE series by its id.

To change just ONE occurrence of a series (not the whole series), use "skip" to drop
that single date, or "move" to change only that occurrence — identify it with
"occurrence" set to that occurrence's scheduled datetime (resolve "this Monday",
"next week's" against CURRENT TIME).

Return a JSON array of operations, each one of:
  {{"op": "create", "title": "<short title>", "start": "<ISO-8601 with offset>", "end": "<ISO-8601 or omit>", "location": "<or omit>", "notes": "<or omit>", "rrule": "<RFC 5545 RRULE or omit>"}}
  {{"op": "reschedule", "id": "<existing event id>", "start": "<new ISO-8601 or omit>", "title": "<or omit>", "location": "<or omit>"}}
  {{"op": "cancel", "id": "<existing event id>"}}
  {{"op": "skip", "id": "<series id>", "occurrence": "<ISO-8601 of the occurrence to drop>"}}
  {{"op": "move", "id": "<series id>", "occurrence": "<ISO-8601 of the original occurrence>", "start": "<new ISO-8601>", "end": "<or omit>", "title": "<or omit>", "location": "<or omit>"}}
Return [] if nothing should change. Output JSON only — no prose, no code fences.

CURRENT TIME: {now}

Already scheduled:
{events}

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
        if isinstance(d, dict)
        and d.get("op") in {"create", "reschedule", "cancel", "skip", "move"}
    ]


def _target_id(settings: Settings, op: dict) -> str | None:
    """Resolve the event an op refers to, by id or a fuzzy title/query fallback."""
    ident = op.get("id") or op.get("query") or op.get("title")
    if not ident:
        return None
    found = store.find_event(settings, str(ident))
    return found.id if found else None


def _log_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    event_id: str,
    op: str,
    summary: str,
    before: store.Event | None,
) -> None:
    if not (thread_id and batch_id and settings.enable_write_confirmation):
        return
    undo.record_write(settings, thread_id, batch_id, event_id, op, summary, before)


def apply_op(
    settings: Settings, op: dict, thread_id: str = "", batch_id: str = ""
) -> str | None:
    """Apply a single parsed operation; return a short log line, or ``None``."""
    kind = op["op"]
    if kind == "create" and op.get("title") and op.get("start"):
        rrule = str(op.get("rrule", "") or "")
        if rrule and not recurrence.validate_rrule(rrule):
            rrule = ""  # keep the event, drop an unparseable rule
        event = store.create_event(
            settings,
            title=str(op["title"]),
            start=str(op["start"]),
            end=str(op.get("end", "") or ""),
            location=str(op.get("location", "") or ""),
            notes=str(op.get("notes", "") or ""),
            rrule=rrule,
        )
        suffix = f" ({recurrence.humanize_rrule(event.rrule)})" if event.rrule else ""
        summary = f"created: {event.title} @ {event.start}{suffix}"
        _log_write(settings, thread_id, batch_id, event.id, "create", summary, None)
        return summary

    if kind == "reschedule":
        target = _target_id(settings, op)
        if target is None:
            return None
        before = store.get_event(settings, target)
        revised = store.update_event(
            settings, target,
            start=op.get("start"),
            end=op.get("end"),
            title=op.get("title"),
            location=op.get("location"),
            notes=op.get("notes"),
        )
        if revised is None:
            return None
        summary = f"rescheduled: {revised.title} @ {revised.start}"
        _log_write(settings, thread_id, batch_id, target, "reschedule", summary, before)
        return summary

    if kind == "cancel":
        target = _target_id(settings, op)
        if target is None:
            return None
        deleted = store.delete_event(settings, target)
        if deleted is None:
            return None
        summary = f"cancelled: {deleted.title}"
        _log_write(settings, thread_id, batch_id, target, "cancel", summary, deleted)
        return summary

    if kind in {"skip", "move"}:
        return _apply_occurrence_op(settings, kind, op, thread_id, batch_id)

    return None


def _apply_occurrence_op(
    settings: Settings, kind: str, op: dict, thread_id: str = "", batch_id: str = ""
) -> str | None:
    """Apply a single-occurrence exception (``skip``/``move``) on a series master."""
    target = _target_id(settings, op)
    when = store.parse_dt(str(op.get("occurrence", "")))
    if target is None or when is None:
        return None
    master = store.get_event(settings, target)
    if master is None or not master.rrule:
        return None  # exceptions only apply to a series
    occurrence = recurrence.resolve_occurrence(master, when, resolve_tz(settings))
    if occurrence is None:
        return None  # no such occurrence in the series
    key = occurrence.isoformat()

    if kind == "skip":
        updated = store.add_exdate(settings, target, key)
        if updated is None:
            return None
        summary = f"skipped: {master.title} on {key}"
        _log_write(settings, thread_id, batch_id, target, "skip", summary, master)
        return summary

    fields = {k: op.get(k) for k in ("start", "end", "title", "location") if op.get(k)}
    if not fields:
        return None
    updated = store.set_override(settings, target, key, fields)
    if updated is None:
        return None
    summary = f"moved: {master.title} {key} -> {fields.get('start', key)}"
    _log_write(settings, thread_id, batch_id, target, "move", summary, master)
    return summary


def update_calendar(
    settings: Settings | None, user_msg: str, assistant_msg: str, thread_id: str = ""
) -> list[str]:
    """Extract and apply calendar operations for one turn (create/reschedule/cancel).

    Intended to run in the background — it makes a second Codex call. Returns a
    short log of what changed. No-ops when ``enable_auto_schedule`` is false.
    When ``thread_id`` is given and ``enable_write_confirmation`` is on, every
    applied op is logged to the undo ledger under one batch and an out-of-band
    confirmation (with an undo hint) is pushed back to that thread.
    """
    settings = settings or get_settings()
    if not settings.enable_auto_schedule:
        return []

    prompt = _SCHEDULE_PROMPT.format(
        now=now(settings).isoformat(timespec="minutes"),
        events=render_events(settings, writer_view(settings), with_ids=True),
        user=user_msg,
        assistant=assistant_msg,
    )
    try:
        raw = run_codex(prompt, settings=settings)
    except Exception:
        logger.exception("calendar extraction (run_codex) failed; skipping this turn")
        return []  # calendar upkeep is best-effort; never break the main flow

    batch_id = uuid.uuid4().hex if thread_id else ""
    applied: list[str] = []
    for op in _parse_ops(raw):
        try:
            result = apply_op(settings, op, thread_id, batch_id)
            if result:
                applied.append(result)
        except Exception:
            logger.exception("failed to apply calendar op: %s", op)

    if applied:
        logger.info("calendar updated: %s", "; ".join(applied))
        if thread_id and settings.enable_write_confirmation:
            try:
                message = "\n".join(applied) + (
                    f'\nReply "undo" within {settings.write_undo_window_minutes} min to revert.'
                )
                notify.deliver_write_confirmation(settings, thread_id, message)
            except Exception:
                logger.exception("failed to push write confirmation for thread %s", thread_id)
    return applied
