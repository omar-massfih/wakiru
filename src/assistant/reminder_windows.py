"""When a dated item's reminder is due — the window math shared by calendar and tasks.

Both reminder paths (:mod:`assistant.calendar.reminders`,
:mod:`assistant.tasks.reminders`) reduce to the same question: given how long
until an item starts (or fell due), which reminder bands cover it right now?
:func:`due_slots` answers it in both of the configured modes:

* **Lead mode** (default): each configured lead L in
  :attr:`Settings.reminder_lead_minutes` is a band ``0 <= remaining <= L``, plus
  one final "starting now" band just after the item begins (within
  :data:`START_GRACE`, keyed as lead 0 so it dedupes independently).
* **Repeat mode** (:attr:`Settings.reminder_repeat_minutes` set): the leads only
  mark when reminders *begin* (their max); the item then re-nudges every
  ``repeat`` minutes until the caller's ``repeat_floor`` is exhausted. Each
  countdown band is a distinct :func:`repeat_slot` integer.

Either way the slots ride the callers' fired-ledger ``lead_minutes`` column, so
each band claims the dedupe ledger exactly once.
"""

from __future__ import annotations

import math
from datetime import timedelta

# How long after its start an item still gets its one "starting now" nudge —
# wide enough to cover ticker jitter and a short restart landing exactly on the
# boundary. The ledger still dedupes, so the at-start band fires at most once.
START_GRACE = timedelta(minutes=5)


def repeat_slot(remaining: timedelta, repeat_minutes: int) -> int:
    """Bucket a countdown into a stable per-interval slot (floored whole minutes).

    Successive ``repeat_minutes``-wide bands map to distinct integers, so each band
    claims the dedupe ledger exactly once (the ledger's ``lead_minutes`` column
    doubles as the slot key). Negative values are overdue bands, used only for
    tasks that keep nagging past their due time.
    """
    return math.floor(remaining.total_seconds() / 60 / repeat_minutes) * repeat_minutes


def due_slots(
    remaining: timedelta,
    leads: list[int],
    repeat: int,
    *,
    repeat_floor: timedelta,
) -> list[int]:
    """The reminder bands covering an item ``remaining`` away, smallest first.

    Returns ``[]`` when nothing is due. In lead mode the list holds every lead
    window the item is currently inside, so the caller can claim them together
    instead of pushing duplicates; in repeat mode it is always a single
    :func:`repeat_slot`. ``repeat_floor`` is how far past the start/due instant
    repeat mode keeps nagging: the calendar allows only the one at-start band,
    tasks their whole overdue window.
    """
    if repeat > 0:
        if not (repeat_floor <= remaining <= timedelta(minutes=max(leads))):
            return []
        return [repeat_slot(remaining, repeat)]
    slots = sorted(
        lead for lead in leads
        if timedelta(0) <= remaining <= timedelta(minutes=lead)
    )
    if not slots and -START_GRACE <= remaining < timedelta(0):
        # The at-start band: one final "starting now" nudge, keyed as lead 0
        # so it dedupes independently of the ahead-of-time leads.
        slots = [0]
    return slots
