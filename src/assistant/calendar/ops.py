"""Calendar writes — the operation set the agent's calendar tools apply.

Each operation arrives as a parsed dict —

* ``create``     — schedule a new event,
* ``reschedule`` — change an existing event's time/details (an in-place update),
* ``cancel``     — remove an event,
* ``skip``/``move`` — single-occurrence exceptions on a recurring series.

Existing events are targeted by id (with a fuzzy title fallback that refuses
ambiguity — cancelling nothing beats cancelling the wrong appointment). Every
applied op is logged to the undo ledger under the turn's batch so "undo"
reverts the whole batch deterministically.
"""

from __future__ import annotations

import logging

from .. import write_ops
from ..config import Settings
from . import recurrence, store, sync, undo
from .context import format_when, overlapping_events, resolve_tz

logger = logging.getLogger(__name__)


def _refuse_ics_mirror(settings: Settings, op: dict, match: store.Event) -> bool:
    """Veto writes to rows mirrored from a read-only ICS feed.

    The feed is the source of truth, and the next pull would silently revert any
    local change — so refuse rather than pretend. (CalDAV-backed rows are
    deliberately NOT refused: is_synced_id is ICS-only, so they fall through and
    their edit is pushed back — see _push_caldav.)
    """
    if not sync.is_synced_id(match.id):
        return False
    logger.warning(
        "calendar %s target %r is synced from an external calendar; skipping",
        op.get("op"), match.title,
    )
    return True


# The find/record_write lambdas resolve store/undo attributes at call time so
# test monkeypatches on those modules keep working (the same rationale as
# write_ledger.LedgerSpec naming its pg adapters by string).
_SPEC: write_ops.WriteOpsSpec[store.Event] = write_ops.WriteOpsSpec(
    kind="calendar",
    noun="events",
    find=lambda settings, ident: store.find_events(settings, ident),
    title_is_new_value=frozenset({"reschedule", "move"}),  # both carry the new title
    record_write=lambda *args: undo.record_write(*args),
    refuse_match=_refuse_ics_mirror,
)


def _target_id(settings: Settings, op: dict) -> str | list[store.Event] | None:
    return write_ops.resolve_target(_SPEC, settings, op)


def _ambiguous_message(settings: Settings, matches: list[store.Event]) -> str:
    shown = ", ".join(
        f'{e.id} ("{e.title}" @ {format_when(settings, e.start)})' for e in matches[:5]
    )
    more = f", +{len(matches) - 5} more" if len(matches) > 5 else ""
    return (
        f"Ambiguous — {len(matches)} events match: {shown}{more}. "
        "Retry with one exact id from Upcoming events."
    )


# Appended to a write-confirmation when the local write landed but the CalDAV push
# didn't — the reconcile pass will retry it.
_UNSYNCED_NOTE = " (not yet synced to your calendar — will retry)"


def _push_caldav(
    settings: Settings,
    event: store.Event | None,
    op: str,
    before: store.Event | None,
) -> bool:
    """Best-effort mirror of a local calendar write to the CalDAV collection.

    Returns ``True`` when the remote is in sync (or there is nothing to push),
    ``False`` when the push failed and was queued to the outbox for reconcile. A
    CalDAV outage must never break the local write, so *every* failure — a transport
    error, an ETag conflict, an unbuildable body — is swallowed here into a queued
    retry. No-op unless ``enable_caldav_write`` is on; ICS-mirrored rows never push.
    """
    if not settings.enable_caldav_write:
        return True
    row = event if event is not None else before
    if row is None or sync.is_synced_id(row.id):
        return True

    from . import outbox, remote

    try:
        if op in ("create", "reschedule") and event is not None:
            try:
                href, etag = remote.upsert(settings, event)
            except Exception:
                outbox.enqueue(settings, event.id, outbox.OP_PUT)
                raise
            store.set_caldav_meta(settings, event.id, href, etag)
            outbox.clear(settings, event.id)
            return True
        if op == "cancel":
            target = before or event
            if target is not None and target.caldav_href:
                try:
                    remote.delete(settings, target.caldav_href, target.caldav_etag or None)
                except Exception:
                    outbox.enqueue(
                        settings, target.id, outbox.OP_DELETE,
                        href=target.caldav_href, etag=target.caldav_etag,
                    )
                    raise
            if target is not None:
                outbox.clear(settings, target.id)
            return True
        return True
    except Exception:
        logger.warning("remote calendar push failed (%s); queued for reconcile", op, exc_info=True)
        return False


def _conflict_note(settings: Settings, event: store.Event) -> str:
    """A `` ⚠ conflicts with <titles>`` suffix if ``event`` overlaps others, else ''.

    Non-blocking — a booking is never refused, only flagged, and the note flows
    into the write-confirmation message the user already receives. Best-effort:
    any failure computing overlaps yields no note rather than breaking the write.
    """
    try:
        conflicts = overlapping_events(settings, event, ignore_id=event.id)
    except Exception:
        logger.exception("overlap check failed for %s", event.id)
        return ""
    if not conflicts:
        return ""
    # Dedupe titles (a recurring series contributes several same-title occurrences).
    titles = ", ".join(dict.fromkeys(c.title for c in conflicts))
    return f" ⚠ conflicts with {titles}"


def _log_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    event_id: str,
    op: str,
    summary: str,
    before: store.Event | None,
) -> None:
    write_ops.log_write(_SPEC, settings, thread_id, batch_id, event_id, op, summary, before)


def apply_op(
    settings: Settings, op: dict, thread_id: str = "", batch_id: str = ""
) -> str | None:
    """Apply a single parsed operation; return a short log line, an
    ambiguous-match message for the model to act on, or ``None`` (nothing
    found/nothing to do)."""
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
        summary = f"created: {event.title} @ {format_when(settings, event.start)}{suffix}"
        summary += _conflict_note(settings, event)
        if not _push_caldav(settings, event, "create", None):
            summary += _UNSYNCED_NOTE
        _log_write(settings, thread_id, batch_id, event.id, "create", summary, None)
        return summary

    if kind == "reschedule":
        target = _target_id(settings, op)
        if isinstance(target, list):
            return _ambiguous_message(settings, target)
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
        summary = f"rescheduled: {revised.title} @ {format_when(settings, revised.start)}"
        summary += _conflict_note(settings, revised)
        if not _push_caldav(settings, revised, "reschedule", before):
            summary += _UNSYNCED_NOTE
        _log_write(settings, thread_id, batch_id, target, "reschedule", summary, before)
        return summary

    if kind == "cancel":
        target = _target_id(settings, op)
        if isinstance(target, list):
            return _ambiguous_message(settings, target)
        if target is None:
            return None
        deleted = store.delete_event(settings, target)
        if deleted is None:
            return None
        summary = f"cancelled: {deleted.title}"
        if not _push_caldav(settings, None, "cancel", deleted):
            summary += _UNSYNCED_NOTE
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
    if isinstance(target, list):
        return _ambiguous_message(settings, target)
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
        summary = f"skipped: {master.title} on {format_when(settings, key)}"
        if not _push_caldav(settings, updated, "reschedule", master):
            summary += _UNSYNCED_NOTE
        _log_write(settings, thread_id, batch_id, target, "skip", summary, master)
        return summary

    fields = {k: str(op[k]) for k in ("start", "end", "title", "location") if op.get(k)}
    if not fields:
        return None
    updated = store.set_override(settings, target, key, fields)
    if updated is None:
        return None
    new_start = str(fields.get("start", key))
    summary = (
        f"moved: {master.title} {format_when(settings, key)}"
        f" -> {format_when(settings, new_start)}"
    )
    if not _push_caldav(settings, updated, "reschedule", master):
        summary += _UNSYNCED_NOTE
    _log_write(settings, thread_id, batch_id, target, "move", summary, master)
    return summary
