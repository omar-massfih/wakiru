"""Proactive birthday reminders for people in the CRM.

Fires one heads-up per person per year when their birthday enters the
``people_birthday_lead_days`` window — so the user has time to plan — even
without the heartbeat enabled (that layer surfaces birthdays too, for a model to
act on; this is the deterministic reminder-channel path). Exactly-once via the
shared fired ledger keyed on ``(person_id, occurrence-date)``, pushed through the
same delivery path calendar and task reminders use. Best-effort and idempotent,
so the in-process ticker and a manual ``POST /reminders/run`` can both drive it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .. import fired_ledger
from ..calendar.context import now
from ..config import Settings, get_settings
from ..notify import deliver_reminder
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


def run_birthday_reminders(settings: Settings | None = None, agent=None) -> list[dict]:
    """Fire every birthday now entering its lead window, exactly once per year.

    No-op when reminders or people are disabled, during quiet hours, or under an
    all-scope mute — the same holds the briefing applies (nothing is claimed, so
    it resumes on the first eligible tick after the hold lifts).
    """
    settings = settings or get_settings()
    if not (settings.enable_reminders and settings.enable_people):
        return []
    current = now(settings)
    from ..memory.profile import in_quiet_hours
    from ..mutes import all_muted

    if in_quiet_hours(settings, current) or all_muted(settings, current):
        return []

    due = due_birthday_reminders(settings, current)
    if not due:
        return []
    fired_at = current.isoformat(timespec="seconds")
    keys = [(r["person_id"], r["occurrence"]) for r in due]
    claimed = fired_ledger.claim(_LEDGER, settings, keys, fired_at, current)
    sent = [due[i] for i in claimed]
    if not sent:
        return []

    # One push per batch, composed in the assistant's own voice; the template
    # text each carries is the fallback, so a model failure still delivers.
    from ..compose import compose_push
    from ..proactive import record_push

    text = compose_push(
        settings,
        instruction=(
            "Compose ONE short, warm heads-up about the upcoming birthday(s) "
            "below, in your own voice, in the user's language. Mention when each "
            "is and gently suggest reaching out or planning something. Reply "
            "with the message only — no preamble, no quotes."
        ),
        facts="\n".join(f"- {r['message']}" for r in sent),
        query=" ".join(r["title"] for r in sent),
        fallback=" ".join(r["message"] for r in sent),
    )
    try:
        delivered = deliver_reminder(settings, {"title": "Birthday", "message": text})
    except Exception:
        # The claim is already committed; delivery is best-effort by design.
        import logging

        logging.getLogger(__name__).exception("birthday reminder delivery failed")
        return sent
    if delivered:
        record_push(agent, settings, f"⏰ {text}")
    return sent
