"""Proactive renewal reminders for tracked subscriptions / bills.

Fires one heads-up per renewal when it enters the
``subscriptions_renewal_lead_days`` window, so a charge never surprises the user
("Spotify renews in 3 days, 129 kr"). Exactly-once via the shared fired ledger
keyed on ``(subscription_id, renewal-date)``, pushed through the same delivery
path calendar/task/birthday reminders use. Best-effort and idempotent.
"""

from __future__ import annotations

from datetime import datetime

from .. import fired_ledger
from ..calendar.context import now
from ..config import Settings, get_settings
from ..notify import deliver_reminder
from . import store
from .context import _amount_str

_LEDGER = fired_ledger.FiredLedgerSpec(
    table="subscription_renewals_fired",
    columns=(("subscription_id", "TEXT"), ("renewal", "TEXT")),
    db_path=lambda settings: settings.subscriptions_db_path,
)


def _fallback(sub: store.Subscription, days: int) -> str:
    when = "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days} days")
    amount = _amount_str(sub)
    tail = f" ({amount})" if amount else ""
    return f"💳 {sub.name} renews {when}{tail}."


def due_renewal_reminders(settings: Settings, current: datetime) -> list[dict]:
    """Subscriptions whose next renewal is within the lead window as of ``current``.

    Pure — no ledger, no delivery. One dict per subscription carrying its
    renewal-date key (stable across the lead window, so the ledger fires once).
    """
    lead = settings.subscriptions_renewal_lead_days
    today = current.date()
    due: list[dict] = []
    for sub in store.list_subscriptions(settings):
        nxt = store.next_renewal(sub, today)
        if nxt is None:
            continue
        days = (nxt - today).days
        if days < 0 or days > lead:
            continue
        due.append(
            {
                "subscription_id": sub.id,
                "renewal": nxt.isoformat(),
                "title": f"{sub.name} renewal",
                "days": days,
                "message": _fallback(sub, days),
            }
        )
    return due


def run_subscription_reminders(settings: Settings | None = None, agent=None) -> list[dict]:
    """Fire every renewal now entering its lead window, exactly once per cycle.

    No-op when reminders or subscriptions are disabled, during quiet hours, or
    under an all-scope mute — the same holds the briefing/birthday reminders apply.
    """
    settings = settings or get_settings()
    if not (settings.enable_reminders and settings.enable_subscriptions):
        return []
    current = now(settings)
    from ..memory.profile import in_quiet_hours
    from ..mutes import all_muted

    if in_quiet_hours(settings, current) or all_muted(settings, current):
        return []

    due = due_renewal_reminders(settings, current)
    if not due:
        return []
    fired_at = current.isoformat(timespec="seconds")
    keys = [(r["subscription_id"], r["renewal"]) for r in due]
    claimed = fired_ledger.claim(_LEDGER, settings, keys, fired_at, current)
    sent = [due[i] for i in claimed]
    if not sent:
        return []

    from ..compose import compose_push
    from ..proactive import record_push

    text = compose_push(
        settings,
        instruction=(
            "Compose ONE short heads-up about the upcoming subscription "
            "renewal(s) below, in your own voice, in the user's language. Note "
            "when each renews and the amount if given, so they can cancel in time "
            "if they want. Reply with the message only — no preamble, no quotes."
        ),
        facts="\n".join(f"- {r['message']}" for r in sent),
        query=" ".join(r["title"] for r in sent),
        fallback=" ".join(r["message"] for r in sent),
    )
    try:
        delivered = deliver_reminder(settings, {"title": "Renewal", "message": text})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("subscription reminder delivery failed")
        return sent
    if delivered:
        record_push(agent, settings, f"⏰ {text}")
    return sent
