"""Proactive reminders — the calendar's signal into the heartbeat.

The read path (:mod:`.context`) and write path (:mod:`.ops`) both only run when the
user chats. Reminders are the missing third path: unprompted nudges ahead of an
event ("Dentist in 1 hour"), surfaced to the heartbeat's situation report rather
than fired directly to a channel — the model judges what to say.

:func:`surface_due` is the entry point. On each heartbeat wake it finds events
entering a configured *lead* window and claims each band exactly once via a
small SQLite dedupe ledger; the claimed reminders ride the situation report.
Which lead windows apply is per event: with importance classification on
(:attr:`Settings.reminder_importance_enabled`, see :mod:`.importance`), critical
events (doctor, flight, exam …) get the long multi-day schedule and everything
else the short :attr:`Settings.reminder_lead_minutes`; with it off, every event
uses :attr:`Settings.reminder_lead_minutes` uniformly. :func:`next_due_at`
tells the wake scheduler when the next band opens, so the heartbeat cadence
doesn't make reminders late.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .. import fired_ledger
from ..config import Settings, get_settings
from ..reminder_windows import START_GRACE, due_slots, next_band_change
from . import importance, recurrence, store
from .context import now

logger = logging.getLogger(__name__)

# The dedupe ledger lives in the same ``calendar.db`` file the store uses.
_LEDGER = fired_ledger.FiredLedgerSpec(
    table="reminders_fired",
    columns=(("event_id", "TEXT"), ("event_start", "TEXT"), ("lead_minutes", "INTEGER")),
    db_path=lambda settings: settings.calendar_db_path,
)


def due_reminders(settings: Settings, current: datetime | None = None) -> list[dict]:
    """Reminders that should fire as of ``current`` (defaults to the assistant's now).

    An event is due when it starts within the next L minutes for one of its lead
    windows L — the tier schedule :mod:`.importance` picks for it, or the uniform
    :attr:`Settings.reminder_lead_minutes` when classification is off — plus one
    final "starting now" band just after it begins (within
    :data:`START_GRACE`, keyed as lead/slot 0 or below so the countdown bands stay
    distinct). An event inside several lead windows at once (e.g.
    booked half an hour ahead with leads of a day and an hour) yields ONE reminder,
    not one per lead: ``lead_minutes`` is the smallest due lead and ``covered_leads``
    lists every lead window the event is currently inside, so the caller can claim
    them together instead of pushing duplicates. Returns one dict per event:
    ``{event_id, title, start, tier, lead_minutes, covered_leads, message}``.
    Pure apart from the importance cache — it does not touch the ledger or
    deliver anything.

    When :attr:`Settings.reminder_repeat_minutes` is set, the leads instead only
    mark when reminders *begin* (their max); the event then re-nudges every
    ``repeat`` minutes until it starts. Each countdown band is a distinct slot
    carried in ``lead_minutes``/``covered_leads``, so the same claim-once ledger
    path fires each band exactly once.
    """
    classify = settings.reminder_importance_enabled
    max_lead = (
        importance.max_lead_minutes(settings)
        if classify
        else max(settings.reminder_lead_minutes, default=0)
    )
    if not max_lead:
        return []
    # Deferred: phrasing imports calendar.context, so a module-level import here
    # would cycle through the calendar package's __init__.
    from ..phrasing import event_reminder_message

    current = current or now(settings)
    horizon = current + timedelta(minutes=max_lead)
    # Expand recurring series so each occurrence is nudged on its own; the ledger
    # keys on the occurrence start, so a weekly standup fires once per week. The
    # window opens START_GRACE early so a just-started event is still seen for
    # its at-start nudge.
    events = recurrence.occurrences_in(settings, current - START_GRACE, horizon)

    tiers: dict[str, str] = {}
    if classify and events:
        try:
            tiers = importance.tiers_for(settings, events)
        except Exception:
            # Classification is advisory: any surprise degrades every event to
            # the normal schedule rather than blocking reminders.
            logger.exception("importance lookup failed; using normal leads")

    repeat = settings.reminder_repeat_minutes
    reminders: list[dict] = []
    for event in events:
        tier = tiers.get(event.id, importance.TIER_NORMAL)
        leads = (
            importance.leads_for(settings, tier)
            if classify
            else settings.reminder_lead_minutes
        )
        if not leads:
            continue
        start = store.parse_dt(event.start)
        if start is None:
            continue
        remaining = start - current
        # In repeat mode the nagging stops at the event's start, bar the ONE
        # "starting now" nudge the grace window allows (the single negative slot).
        slots = due_slots(
            remaining, leads, repeat,
            repeat_floor=-min(START_GRACE, timedelta(minutes=repeat)),
        )
        if not slots:
            continue
        reminders.append(
            {
                "event_id": event.id,
                "title": event.title,
                "start": event.start,
                "tier": tier,
                "lead_minutes": slots[0],
                "covered_leads": slots,
                "message": event_reminder_message(
                    settings, event.id, event.title, event.start, remaining, slots[0]
                ),
            }
        )
    return reminders


def surface_due(settings: Settings | None = None, current: datetime | None = None) -> list[dict]:
    """Claim every reminder now due, exactly once, and return it for the heartbeat.

    Idempotent: each due band is claimed with an atomic ``INSERT OR IGNORE`` on
    the ledger, so a band already surfaced (by an earlier wake or an overlapping
    manual call) is silently skipped. A rescheduled event surfaces afresh
    because the ledger key includes the event's start. No-op returning ``[]``
    when ``enable_reminders`` is false. Quiet hours are the caller's hold —
    the heartbeat gathers nothing during them, so nothing is claimed.
    """
    settings = settings or get_settings()
    if not settings.enable_reminders:
        return []

    current = current or now(settings)
    # Compute the due list first, with its own (store) connections, so the ledger
    # write transaction inside claim_due never overlaps a nested connection to the
    # same DB.
    due = due_reminders(settings, current)
    return fired_ledger.claim_due(
        _LEDGER,
        settings,
        due,
        current=current,
        kind="event",
        key_fields=("event_id", "start"),
        pg_claim="claim_calendar_reminders",
    )


def next_due_at(settings: Settings, current: datetime, until: datetime) -> list[datetime]:
    """When upcoming events' next reminder bands open, within ``(current, until]``.

    A pure read for the heartbeat's wake scheduler: cached importance tiers
    only (never a classification call), the same window math ``due_reminders``
    uses. An unclassified event is scanned with the normal schedule — at worst
    it is woken for on the base cadence and graded on that wake.
    """
    if not settings.enable_reminders:
        return []
    classify = settings.reminder_importance_enabled
    max_lead = (
        importance.max_lead_minutes(settings)
        if classify
        else max(settings.reminder_lead_minutes, default=0)
    )
    if not max_lead:
        return []
    events = recurrence.occurrences_in(
        settings, current, until + timedelta(minutes=max_lead)
    )
    tiers = importance.cached_tiers(settings, events) if classify and events else {}
    repeat = settings.reminder_repeat_minutes
    openings: list[datetime] = []
    for event in events:
        leads = (
            importance.leads_for(settings, tiers.get(event.id, importance.TIER_NORMAL))
            if classify
            else settings.reminder_lead_minutes
        )
        if not leads:
            continue
        start = store.parse_dt(event.start)
        if start is None:
            continue
        change = next_band_change(
            start - current, leads, repeat,
            repeat_floor=-min(START_GRACE, timedelta(minutes=repeat or 0)),
        )
        if change is not None and current + change <= until:
            openings.append(current + change)
    return openings
