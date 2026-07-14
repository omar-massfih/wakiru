"""Proactive reminders — the calendar's wall-clock output path.

The read path (:mod:`.context`) and write path (:mod:`.ops`) both only run when the
user chats. Reminders are the missing third path: unprompted nudges ahead of an
event ("Dentist in 1 hour"), driven by a wall-clock ticker rather than chat traffic.

:func:`run_reminders` is the entry point. On each call it finds events entering a
configured *lead* window (:attr:`Settings.reminder_lead_minutes`), fires each one
exactly once via a small SQLite dedupe ledger, and pushes it through
:func:`assistant.notify.deliver_reminder`. It is best-effort and idempotent, so the
in-process ticker and a manual ``POST /reminders/run`` can both drive it safely.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from .. import fired_ledger
from ..config import Settings, get_settings
from ..notify import deliver_reminder
from . import recurrence, store
from .context import now

logger = logging.getLogger(__name__)

# The dedupe ledger lives in the same ``calendar.db`` file the store uses.
_LEDGER = fired_ledger.FiredLedgerSpec(
    table="reminders_fired",
    columns=(("event_id", "TEXT"), ("event_start", "TEXT"), ("lead_minutes", "INTEGER")),
    db_path=lambda settings: settings.calendar_db_path,
)


def _repeat_slot(remaining: timedelta, repeat_minutes: int) -> int:
    """Bucket a countdown into a stable per-interval slot (floored whole minutes).

    Successive ``repeat_minutes``-wide bands map to distinct integers, so each band
    claims the dedupe ledger exactly once (the ledger's ``lead_minutes`` column
    doubles as the slot key). Negative values are overdue bands, used only for
    tasks that keep nagging past their due time.
    """
    return math.floor(remaining.total_seconds() / 60 / repeat_minutes) * repeat_minutes


def due_reminders(settings: Settings, current: datetime | None = None) -> list[dict]:
    """Reminders that should fire as of ``current`` (defaults to the assistant's now).

    An event is due when it starts within the next L minutes for a configured lead
    L (and is not already past). An event inside several lead windows at once (e.g.
    booked half an hour ahead with leads of a day and an hour) yields ONE reminder,
    not one per lead: ``lead_minutes`` is the smallest due lead and ``covered_leads``
    lists every lead window the event is currently inside, so the caller can claim
    them together instead of pushing duplicates. Returns one dict per event:
    ``{event_id, title, start, lead_minutes, covered_leads, message}``. Pure — it
    does not touch the ledger or deliver anything.

    When :attr:`Settings.reminder_repeat_minutes` is set, the leads instead only
    mark when reminders *begin* (their max); the event then re-nudges every
    ``repeat`` minutes until it starts. Each countdown band is a distinct slot
    carried in ``lead_minutes``/``covered_leads``, so the same claim-once ledger
    path fires each band exactly once.
    """
    leads = settings.reminder_lead_minutes
    if not leads:
        return []
    # Deferred: phrasing imports calendar.context, so a module-level import here
    # would cycle through the calendar package's __init__.
    from ..phrasing import event_reminder_message

    current = current or now(settings)
    horizon = current + timedelta(minutes=max(leads))
    # Expand recurring series so each occurrence is nudged on its own; the ledger
    # keys on the occurrence start, so a weekly standup fires once per week.
    events = recurrence.occurrences_in(settings, current, horizon)

    repeat = settings.reminder_repeat_minutes
    max_lead = max(leads)
    reminders: list[dict] = []
    for event in events:
        start = store.parse_dt(event.start)
        if start is None:
            continue
        remaining = start - current
        if repeat > 0:
            # Repeat mode: begin at the outermost lead, then re-notify every
            # `repeat` minutes until the event starts. Each countdown band is a
            # distinct slot, claimed (and pushed) exactly once. Nothing fires once
            # the event has started (remaining < 0).
            if not (timedelta(0) <= remaining <= timedelta(minutes=max_lead)):
                continue
            slot = _repeat_slot(remaining, repeat)
            reminders.append(
                {
                    "event_id": event.id,
                    "title": event.title,
                    "start": event.start,
                    "lead_minutes": slot,
                    "covered_leads": [slot],
                    "message": event_reminder_message(
                        settings, event.id, event.title, event.start, remaining, slot
                    ),
                }
            )
            continue
        due_leads = sorted(
            lead for lead in leads
            if timedelta(0) <= remaining <= timedelta(minutes=lead)
        )
        if due_leads:
            reminders.append(
                {
                    "event_id": event.id,
                    "title": event.title,
                    "start": event.start,
                    "lead_minutes": due_leads[0],
                    "covered_leads": due_leads,
                    "message": event_reminder_message(
                        settings, event.id, event.title, event.start, remaining, due_leads[0]
                    ),
                }
            )
    return reminders


def run_reminders(settings: Settings | None = None, agent=None) -> list[dict]:
    """Fire every reminder now due, exactly once, and return what was sent.

    Best-effort and idempotent: each due reminder is claimed with an atomic
    ``INSERT OR IGNORE`` on the ledger, so a reminder already fired (by an earlier
    tick or an overlapping manual call) is silently skipped. A rescheduled event
    fires afresh because the ledger key includes the event's start. No-op returning
    ``[]`` when ``enable_reminders`` is false. With ``agent`` given, each delivered
    reminder is also recorded into the authorized chats' working memory (see
    :mod:`assistant.proactive`), so conversations know what was pushed.
    """
    settings = settings or get_settings()
    if not settings.enable_reminders:
        return []

    current = now(settings)
    # Honor stated quiet hours (profile): nothing is claimed, so a reminder whose
    # window survives the night fires on the first tick after quiet ends.
    from ..memory.profile import in_quiet_hours

    if in_quiet_hours(settings, current):
        return []
    fired_at = current.isoformat(timespec="seconds")
    # Compute the due list first, with its own (store) connections, so the ledger
    # write transaction below never overlaps a nested connection to the same DB.
    due = due_reminders(settings, current)
    # Honor active mutes (the agent's "stop nudging me about this" switch) the
    # same way quiet hours are honored: filtered before the claim, so nudges
    # resume on the first tick after a mute expires.
    from ..mutes import filter_muted

    due = filter_muted(settings, due, current, "event")

    # Claim first, commit, deliver after: delivery is network I/O (webhook POST,
    # a Telegram send per chat) and must not run inside the ledger's write
    # transaction, where it would hold SQLite's single writer slot past other
    # writers' busy timeouts. The cost is at-most-once delivery: a claimed
    # reminder whose push fails is not retried.
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        sent = storage_postgres.claim_calendar_reminders(settings, due, fired_at, current)
    else:
        # Claim every lead window each event is currently inside, so the larger
        # leads can't fire a duplicate nudge on a later tick; a reminder is sent
        # when any of its windows was newly claimed.
        keys = [
            (reminder["event_id"], reminder["start"], lead)
            for reminder in due
            for lead in reminder["covered_leads"]
        ]
        owner = [
            index
            for index, reminder in enumerate(due)
            for _ in reminder["covered_leads"]
        ]
        claimed = fired_ledger.claim(_LEDGER, settings, keys, fired_at, current)
        sent_indexes = sorted({owner[key_index] for key_index in claimed})
        sent = [due[index] for index in sent_indexes]

    from ..proactive import record_push

    for reminder in sent:
        try:
            delivered = deliver_reminder(settings, reminder)
        except Exception:
            # The claim is already committed; a push that blows up must not
            # take the rest of this batch down with it.
            logger.exception("reminder delivery failed: %s", reminder["message"])
            continue
        if delivered:
            # Recorded with the same ⏰ prefix the chat channels show, so the
            # thread's history matches what the user actually saw.
            record_push(agent, settings, f"⏰ {reminder['message']}")

    if sent:
        logger.info("fired %d reminder(s): %s", len(sent), "; ".join(r["message"] for r in sent))
    return sent
