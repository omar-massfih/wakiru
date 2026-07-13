"""Proactive reminders for tasks with a due date.

The task equivalent of :mod:`assistant.calendar.reminders`, but simpler — a task
has a single ``due`` instant with no recurrence. On each call
:func:`run_task_reminders` finds open, dated tasks entering a configured *lead*
window (:attr:`Settings.reminder_lead_minutes`, shared with the calendar), fires
each exactly once via a small SQLite dedupe ledger in ``tasks.db``, and pushes it
through :func:`assistant.notify.deliver_reminder`. Best-effort and idempotent, so
the in-process ticker and a manual ``POST /reminders/run`` can both drive it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .. import fired_ledger
from ..calendar.context import now
from ..calendar.reminders import _humanize, _humanize_ago, _repeat_slot
from ..calendar.store import parse_dt
from ..config import Settings, get_settings
from ..notify import deliver_reminder
from . import store

logger = logging.getLogger(__name__)

# The dedupe ledger lives in the same ``tasks.db`` file the store uses.
_LEDGER = fired_ledger.FiredLedgerSpec(
    table="task_reminders_fired",
    columns=(("task_id", "TEXT"), ("due", "TEXT"), ("lead_minutes", "INTEGER")),
    db_path=lambda settings: settings.tasks_db_path,
)


def due_task_reminders(settings: Settings, current: datetime | None = None) -> list[dict]:
    """Reminders that should fire as of ``current`` for open, dated tasks.

    A task is due when its ``due`` falls within the next L minutes for a configured
    lead L (and is not already past). Pure — it doesn't touch the ledger or deliver.
    Returns one dict per task: ``{task_id, title, due, lead_minutes, covered_leads,
    message}`` — the same shape the calendar's ``due_reminders`` returns, so the
    delivery path is shared.

    When :attr:`Settings.reminder_repeat_minutes` is set, a dated task instead
    re-nudges every ``repeat`` minutes from its outermost lead onward, and keeps
    nagging past its due time (up to ``reminder_overdue_max_minutes``) until it is
    marked done — ``store.list_tasks`` only returns open tasks, so completing one
    stops the nagging on the next tick.
    """
    leads = settings.reminder_lead_minutes
    if not leads:
        return []

    current = current or now(settings)
    repeat = settings.reminder_repeat_minutes
    max_lead = max(leads)
    overdue_floor = timedelta(minutes=-settings.reminder_overdue_max_minutes)
    reminders: list[dict] = []
    for task in store.list_tasks(settings):  # open tasks only
        due = parse_dt(task.due)
        if due is None:
            continue
        remaining = due - current
        if repeat > 0:
            # Repeat mode: nudge every `repeat` minutes from the outermost lead
            # onward, and keep nagging while overdue until the task is done or the
            # overdue window is exhausted. Each countdown band is a distinct slot.
            if not (overdue_floor <= remaining <= timedelta(minutes=max_lead)):
                continue
            slot = _repeat_slot(remaining, repeat)
            if remaining < timedelta(0):
                message = f"Task overdue: {task.title} ({_humanize_ago(-remaining)})"
            else:
                message = f"Task due: {task.title} {_humanize(remaining)}"
            reminders.append(
                {
                    "task_id": task.id,
                    "title": task.title,
                    "due": task.due,
                    "lead_minutes": slot,
                    "covered_leads": [slot],
                    "message": message,
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
                    "task_id": task.id,
                    "title": task.title,
                    "due": task.due,
                    "lead_minutes": due_leads[0],
                    "covered_leads": due_leads,
                    "message": f"Task due: {task.title} {_humanize(remaining)}",
                }
            )
    return reminders


def run_task_reminders(settings: Settings | None = None, agent=None) -> list[dict]:
    """Fire every due-task reminder now due, exactly once, and return what was sent.

    Same claim-first / deliver-after discipline (and the same loop-in recording
    via ``agent``) as :func:`assistant.calendar.reminders.run_reminders`. No-op
    returning ``[]`` when reminders or tasks are disabled.
    """
    settings = settings or get_settings()
    if not (settings.enable_reminders and settings.enable_tasks):
        return []

    current = now(settings)
    # Same quiet-hours hold as calendar reminders: nothing is claimed, so the
    # nag resumes on the first tick after quiet ends (within the overdue bound).
    from ..memory.profile import in_quiet_hours

    if in_quiet_hours(settings, current):
        return []
    fired_at = current.isoformat(timespec="seconds")
    due = due_task_reminders(settings, current)
    # Same mute hold as calendar reminders: filtered before the claim.
    from ..mutes import filter_muted

    due = filter_muted(settings, due, current, "task")

    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        sent = storage_postgres.claim_task_reminders(settings, due, fired_at, current)
    else:
        keys = [
            (reminder["task_id"], reminder["due"], lead)
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
            logger.exception("task reminder delivery failed: %s", reminder["message"])
            continue
        if delivered:
            record_push(agent, settings, f"⏰ {reminder['message']}")

    if sent:
        logger.info(
            "fired %d task reminder(s): %s", len(sent), "; ".join(r["message"] for r in sent)
        )
    return sent
