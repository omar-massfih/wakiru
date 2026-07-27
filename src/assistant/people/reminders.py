"""Proactive birthday reminders for people in the CRM.

Surfaces one heads-up per person per year when their birthday enters the
``people_birthday_lead_days`` window — so the user has time to plan — into the
heartbeat's situation report, where the model judges what to say. Exactly-once
via the shared fired ledger keyed on ``(person_id, occurrence-date)``, the same
claim-once discipline calendar and task reminders use.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .. import fired_ledger
from ..calendar.context import now
from ..config import Settings, get_settings
from . import store
from .context import days_until_birthday

# The dedupe ledger lives in the same ``people.db`` file the store uses. Under
# Postgres, fired_ledger.claim routes to the generic assistant_fired_ledger
# (keyed by table name), so no per-domain Postgres table is needed.
_LEDGER = fired_ledger.FiredLedgerSpec(
    table="person_birthdays_fired",
    columns=(("person_id", "TEXT"), ("occurrence", "TEXT")),
    db_path=lambda settings: settings.people_db_path,
)


def _when_phrase(days: int) -> str:
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    return f"in {days} days"


def _fallback(name: str, relationship: str, days: int) -> str:
    who = name + (f" ({relationship})" if relationship else "")
    return f"🎂 {who}'s birthday is {_when_phrase(days)}."


def due_birthday_reminders(settings: Settings, current: datetime) -> list[dict]:
    """People whose birthday is within the lead window as of ``current``.

    Pure — no ledger, no delivery. One dict per person carrying its
    occurrence-date key (the actual birthday date, stable across the whole lead
    window, so the ledger fires the heads-up once per year).
    """
    lead = settings.people_birthday_lead_days
    due: list[dict] = []
    for person in store.list_people(settings):
        days = days_until_birthday(person, current)
        if days is None or days > lead:
            continue
        occurrence = (current.date() + timedelta(days=days)).isoformat()
        due.append(
            {
                "person_id": person.id,
                "occurrence": occurrence,
                "title": f"{person.name}'s birthday",
                "days": days,
                "message": _fallback(person.name, person.relationship, days),
            }
        )
    return due


def surface_due(settings: Settings | None = None, current: datetime | None = None) -> list[dict]:
    """Claim every birthday now entering its lead window, exactly once per year.

    Returns the claimed heads-ups for the heartbeat's situation report. No-op
    when reminders or people are disabled. Quiet hours and all-scope mutes are
    the caller's hold — the heartbeat gathers nothing during them, so nothing
    is claimed and the heads-up resumes on the first eligible wake.
    """
    settings = settings or get_settings()
    if not (settings.enable_reminders and settings.enable_people):
        return []
    current = current or now(settings)

    due = due_birthday_reminders(settings, current)
    if not due:
        return []
    fired_at = current.isoformat(timespec="seconds")
    keys = [(r["person_id"], r["occurrence"]) for r in due]
    claimed = fired_ledger.claim(_LEDGER, settings, keys, fired_at, current)
    return [due[i] for i in claimed]
