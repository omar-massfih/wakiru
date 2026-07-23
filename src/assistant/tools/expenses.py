"""Expense tools — log one-off spending and roll a month up by category."""
from __future__ import annotations

from ._base import _ISO, ToolContext, ToolSpec, _params


def _log_expense(ctx: ToolContext, **args: object) -> str:
    from ..expenses import store

    entry = store.log_entry(
        ctx.settings,
        amount=args.get("amount", 0),
        currency=str(args.get("currency", "") or ""),
        category=str(args.get("category", "") or ""),
        note=str(args.get("note", "") or ""),
        on=str(args.get("on", "") or ""),
    )
    if entry is None:
        return "Tool failed: a positive amount is required."
    cur = f" {entry.currency}" if entry.currency else ""
    num = int(entry.amount) if entry.amount == int(entry.amount) else round(entry.amount, 2)
    cat = f" on {entry.category}" if entry.category else ""
    return f"Logged {num}{cur}{cat} ({entry.spent_on})"


def _expense_summary(ctx: ToolContext, **args: object) -> str:
    from ..expenses import context, store

    today = store._today(ctx.settings)
    month = store.parse_month(str(args.get("month", "") or "")) or today.isoformat()[:7]
    return context.month_summary(ctx.settings, month, today)


def _remove_expense(ctx: ToolContext, **args: object) -> str:
    from ..expenses import store

    entry_id = str(args.get("id", "")).strip()
    if not entry_id:
        return "Tool failed: an entry id is required (see expense_summary)."
    removed = store.delete_entry(ctx.settings, entry_id)
    if removed is None:
        return f"No expense with id {entry_id!r}."
    cur = f" {removed.currency}" if removed.currency else ""
    num = int(removed.amount) if removed.amount == int(removed.amount) else round(removed.amount, 2)
    return f"Removed the {num}{cur} expense from {removed.spent_on}."


def _set_budget(ctx: ToolContext, **args: object) -> str:
    from ..expenses import store

    budget = store.set_budget(
        ctx.settings,
        category=str(args.get("category", "") or ""),
        amount=args.get("amount", 0),
        currency=str(args.get("currency", "") or ""),
    )
    if budget is None:
        return "Tool failed: a positive amount is required."
    cur = f" {budget.currency}" if budget.currency else ""
    num = int(budget.amount) if budget.amount == int(budget.amount) else round(budget.amount, 2)
    label = budget.category or "overall spending"
    return f"Budget set: {label} — {num}{cur} per month."


def _remove_budget(ctx: ToolContext, **args: object) -> str:
    from ..expenses import store

    category = str(args.get("category", "") or "")
    removed = store.remove_budget(ctx.settings, category)
    if removed is None:
        label = store.normalize_budget_category(category) or "overall"
        return f"No {label} budget is set."
    return f"Removed the {removed.category or 'overall'} budget."


def _expense_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "log_expense",
            "Record money the user spent — \"250 kr on groceries\", \"coffee, "
            "45\". Capture the amount (required), and the currency, a short "
            "category, and a note when given; use it whenever they mention "
            "spending or buying something.",
            _params(
                {
                    "amount": ("string", "How much, e.g. \"250\" or \"49.90\""),
                    "currency": ("string", "The currency as they said it, e.g. \"kr\", \"EUR\""),
                    "category": ("string", "A short category, e.g. \"groceries\", \"transport\""),
                    "note": ("string", "What it was, e.g. \"Rema\", \"train to Bergen\""),
                    "on": ("string", f"When, {_ISO} or YYYY-MM-DD; omit for today"),
                },
                ["amount"],
            ),
            _log_expense,
        ),
        ToolSpec(
            "expense_summary",
            "Roll up the user's logged expenses for a month — total per "
            "currency, breakdown by category, spend against any budgets, and "
            "recent entries with ids. Defaults to the current month; pass "
            "YYYY-MM for another.",
            _params({"month": ("string", "The month as YYYY-MM; omit for this month")}, []),
            _expense_summary,
        ),
        ToolSpec(
            "remove_expense",
            "Delete a single logged expense by its id (from expense_summary) — "
            "for correcting a mistaken log.",
            _params({"id": ("string", "Exact entry id")}, ["id"]),
            _remove_expense,
        ),
        ToolSpec(
            "set_budget",
            "Set (or update) a monthly spending budget — \"keep groceries "
            "under 3000 kr\". Leave the category empty for an overall monthly "
            "budget; expense_summary then reports spend against it.",
            _params(
                {
                    "amount": ("string", "The monthly cap, e.g. \"3000\""),
                    "category": ("string", "The category to cap, e.g. \"groceries\"; omit for overall"),
                    "currency": ("string", "The currency, e.g. \"kr\"; omit to count all currencies"),
                },
                ["amount"],
            ),
            _set_budget,
        ),
        ToolSpec(
            "remove_budget",
            "Drop a monthly spending budget by its category (omit for the "
            "overall budget).",
            _params({"category": ("string", "The budgeted category; omit for overall")}, []),
            _remove_budget,
        ),
    ]
