"""Proactive reminders for tasks with a due date.

The task equivalent of :mod:`assistant.calendar.reminders`, but simpler — a task
has a single ``due`` instant (a recurring task's due rolls forward on
completion, re-arming these reminders for the next occurrence). On each
heartbeat wake :func:`surface_due` finds open, dated tasks entering a
configured *lead* window (:attr:`Settings.reminder_lead_minutes`, shared with
the calendar), claims each band exactly once via a small SQLite dedupe ledger
in ``tasks.db``, and hands it to the heartbeat's situation report — the model
judges what to say. :func:`next_due_at` tells the wake scheduler when the next
band opens.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .. import fired_ledger
from ..calendar.context import now
from ..calendar.store import parse_dt
from ..config import Settings, get_settings
from ..reminder_windows import START_GRACE, due_slots, next_band_change
from . import store

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
    # Deferred for the same reason as in calendar.reminders: phrasing imports
    # calendar.context, and a top-level import would cycle through that package.
    from ..phrasing import task_reminder_message

    current = current or now(settings)
    repeat = settings.reminder_repeat_minutes
    reminders: list[dict] = []
    for task in store.list_tasks(settings):  # open tasks only
        due = parse_dt(task.due)
        if due is None:
            continue
        remaining = due - current
        slots = due_slots(
            remaining, leads, repeat, repeat_floor=_overdue_floor(settings, task, repeat)
        )
        if not slots:
            continue
        reminders.append(
            {
                "task_id": task.id,
                "title": task.title,
                "due": task.due,
                "lead_minutes": slots[0],
                "covered_leads": slots,
                "message": task_reminder_message(
                    settings, task.id, task.title, task.due, remaining, slots[0]
                ),
            }
        )
    return reminders


def _overdue_floor(settings: Settings, task, repeat: int) -> timedelta:
    """How far past its due instant a task keeps nagging (as a negative delta)."""
    if task.notify_only:
        # A one-time informational nudge: fire the leads and one "now" band,
        # then go silent — never chase it overdue, whatever the repeat config.
        return -START_GRACE
    # In repeat mode the nagging continues while overdue, until the task
    # is done or the overdue window is exhausted — bounded by *both* a
    # time window and a count of re-nudges, so a forgotten task can't nag
    # dozens of times before the day is out.
    overdue_minutes = settings.reminder_overdue_max_minutes
    if repeat > 0 and settings.reminder_overdue_max_nudges > 0:
        overdue_minutes = min(
            overdue_minutes, settings.reminder_overdue_max_nudges * repeat
        )
    return timedelta(minutes=-overdue_minutes)


def surface_due(settings: Settings | None = None, current: datetime | None = None) -> list[dict]:
    """Claim every due-task reminder now due, exactly once, for the heartbeat.

    Same claim-once discipline as :func:`assistant.calendar.reminders.surface_due`
    — both are thin wrappers over :func:`assistant.fired_ledger.claim_due`.
    No-op returning ``[]`` when reminders or tasks are disabled. Quiet hours
    are the caller's hold.
    """
    settings = settings or get_settings()
    if not (settings.enable_reminders and settings.enable_tasks):
        return []

    current = current or now(settings)
    due = due_task_reminders(settings, current)
    return fired_ledger.claim_due(
        _LEDGER,
        settings,
        due,
        current=current,
        kind="task",
        key_fields=("task_id", "due"),
        pg_claim="claim_task_reminders",
    )


def next_due_at(settings: Settings, current: datetime, until: datetime) -> list[datetime]:
    """When open tasks' next reminder bands open, within ``(current, until]``.

    A pure read for the heartbeat's wake scheduler — the same window math
    ``due_task_reminders`` uses, including the overdue repeat bands.
    """
    if not (settings.enable_reminders and settings.enable_tasks):
        return []
    leads = settings.reminder_lead_minutes
    if not leads:
        return []
    repeat = settings.reminder_repeat_minutes
    openings: list[datetime] = []
    for task in store.list_tasks(settings):  # open tasks only
        due = parse_dt(task.due)
        if due is None:
            continue
        change = next_band_change(
            due - current, leads, repeat,
            repeat_floor=_overdue_floor(settings, task, repeat),
        )
        if change is not None and current + change <= until:
            openings.append(current + change)
    return openings
