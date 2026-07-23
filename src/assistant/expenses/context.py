"""Monthly rollup + rendering over the expense log — shared by the expense
tools, the GET /expenses endpoint, and the briefing's start-of-month section.
"""

from __future__ import annotations

from datetime import date, timedelta

from ..config import Settings
from . import store
from .store import ExpenseEntry


def _num(value: float) -> str:
    return str(int(value)) if value == int(value) else str(round(value, 2))


def _amount_str(entry: ExpenseEntry) -> str:
    cur = f" {entry.currency}" if entry.currency else ""
    return f"{_num(entry.amount)}{cur}"


def totals_by_currency(entries: list[ExpenseEntry]) -> dict[str, float]:
    """Total spend per currency (an empty currency groups under "?")."""
    totals: dict[str, float] = {}
    for e in entries:
        cur = e.currency or "?"
        totals[cur] = totals.get(cur, 0.0) + e.amount
    return totals


def totals_by_category(entries: list[ExpenseEntry]) -> dict[str, dict[str, float]]:
    """Per-category spend, each category broken down per currency, largest
    category first (by its biggest single-currency total)."""
    cats: dict[str, dict[str, float]] = {}
    for e in entries:
        per_cur = cats.setdefault(e.category or "uncategorized", {})
        cur = e.currency or "?"
        per_cur[cur] = per_cur.get(cur, 0.0) + e.amount
    return dict(
        sorted(cats.items(), key=lambda kv: max(kv[1].values()), reverse=True)
    )


def _render_totals(totals: dict[str, float]) -> str:
    return " + ".join(
        f"{_num(v)} {cur}" if cur != "?" else _num(v)
        for cur, v in sorted(totals.items())
    )


def month_summary(settings: Settings, month: str, today: date) -> str:
    """A detailed rollup for one month: totals, categories, recent entries with
    ids (so the model can correct a mis-log). ``month`` is ``YYYY-MM``."""
    entries = store.list_entries(settings, month=month)
    if not entries:
        return f"No expenses logged for {month}."
    label = "month to date" if month == today.isoformat()[:7] else "full month"
    lines = [f"Expenses for {month} ({label}):"]
    lines.append(f"  total: {_render_totals(totals_by_currency(entries))}"
                 f" across {len(entries)} expense(s)")
    lines.append("  by category:")
    for cat, per_cur in totals_by_category(entries).items():
        lines.append(f"    - {cat}: {_render_totals(per_cur)}")
    lines.append("  recent:")
    for e in entries[:8]:
        cat = f" {e.category}" if e.category else ""
        note = f" ({e.note})" if e.note else ""
        lines.append(f"    - {e.spent_on} {_amount_str(e)}{cat}{note}  [id: {e.id}]")
    return "\n".join(lines)


def briefing_expenses(settings: Settings, today: date) -> str:
    """The briefing's monthly rollup: on the 1st, last month's spending — total
    and top categories, no ids. Empty on every other day, and when last month
    has no entries, so a quiet log never pads the briefing."""
    if today.day != 1:
        return ""
    last_month = (today - timedelta(days=1)).isoformat()[:7]
    entries = store.list_entries(settings, month=last_month)
    if not entries:
        return ""
    lines = [f"## Last month's spending ({last_month})"]
    lines.append(f"Total: {_render_totals(totals_by_currency(entries))}"
                 f" across {len(entries)} logged expense(s).")
    top = list(totals_by_category(entries).items())[:5]
    for cat, per_cur in top:
        lines.append(f"- {cat}: {_render_totals(per_cur)}")
    return "\n".join(lines)
