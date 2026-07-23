"""Rendering + cost rollup for subscriptions — shared by the list tool and the
renewal reminders."""

from __future__ import annotations

from datetime import date

from ..config import Settings
from .store import Subscription, monthly_amount, next_renewal


def _amount_str(sub: Subscription) -> str:
    if not sub.amount:
        return ""
    cur = f" {sub.currency}" if sub.currency else ""
    # Trim a trailing .0 so whole amounts read cleanly (99 kr, not 99.0 kr).
    n = int(sub.amount) if sub.amount == int(sub.amount) else round(sub.amount, 2)
    return f"{n}{cur}"


def _renewal_str(sub: Subscription, today: date) -> str:
    nxt = next_renewal(sub, today)
    if nxt is None:
        return ""
    days = (nxt - today).days
    when = "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days} days")
    return f"renews {when} ({nxt.isoformat()})"


def render_subscription(sub: Subscription, today: date, with_id: bool = True) -> str:
    line = f"- {sub.name}"
    amount = _amount_str(sub)
    if amount:
        line += f" — {amount}/{sub.cadence.rstrip('ly')}" if sub.cadence else f" — {amount}"
    renewal = _renewal_str(sub, today)
    if renewal:
        line += f"; {renewal}"
    if sub.notes:
        line += f" [{sub.notes}]"
    if with_id:
        line += f"  [id: {sub.id}]"
    return line


def monthly_totals(subs: list[Subscription]) -> dict[str, float]:
    """Total normalized monthly spend per currency (only priced, known-cadence subs)."""
    totals: dict[str, float] = {}
    for sub in subs:
        m = monthly_amount(sub)
        if m is None or not sub.amount:
            continue
        cur = sub.currency or "?"
        totals[cur] = totals.get(cur, 0.0) + m
    return totals


def render_totals(totals: dict[str, float]) -> str:
    if not totals:
        return ""
    parts = [f"{round(v, 2)} {cur}" for cur, v in sorted(totals.items())]
    return "Estimated monthly spend: " + ", ".join(parts)


def rollup(settings: Settings, subs: list[Subscription], today: date) -> str:
    """The full list block: each subscription plus the monthly-spend total."""
    if not subs:
        return "No subscriptions tracked."
    lines = [render_subscription(s, today) for s in subs]
    total = render_totals(monthly_totals(subs))
    if total:
        lines.append(total)
    return "\n".join(lines)
