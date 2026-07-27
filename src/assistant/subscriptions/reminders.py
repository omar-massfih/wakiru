"""Proactive renewal reminders for tracked subscriptions / bills.

Surfaces one heads-up per renewal when it enters the
``subscriptions_renewal_lead_days`` window, so a charge never surprises the user
("Spotify renews in 3 days, 129 kr"). Exactly-once via the shared fired ledger
keyed on ``(subscription_id, renewal-date)``, handed to the heartbeat's
situation report the same way calendar/task/birthday reminders are.
"""

from __future__ import annotations

from datetime import datetime

from .. import fired_ledger
from ..calendar.context import now
from ..config import Settings, get_settings
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


def surface_due(settings: Settings | None = None, current: datetime | None = None) -> list[dict]:
    """Claim every renewal now entering its lead window, exactly once per cycle.

    Returns the claimed heads-ups for the heartbeat's situation report. No-op
    when reminders or subscriptions are disabled. Quiet hours and all-scope
    mutes are the caller's hold — nothing is claimed during them.
    """
    settings = settings or get_settings()
    if not (settings.enable_reminders and settings.enable_subscriptions):
        return []
    current = current or now(settings)

    due = due_renewal_reminders(settings, current)
    if not due:
        return []
    fired_at = current.isoformat(timespec="seconds")
    keys = [(r["subscription_id"], r["renewal"]) for r in due]
    claimed = fired_ledger.claim(_LEDGER, settings, keys, fired_at, current)
    return [due[i] for i in claimed]
