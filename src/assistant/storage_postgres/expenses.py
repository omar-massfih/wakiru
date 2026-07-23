"""Expense-log table for the Postgres backend (twin of assistant.expenses.store)."""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect

_COLS = "id, amount, currency, category, note, spent_on, created"


def ensure_expenses_schema(settings: Settings) -> None:
    if _schema_done(settings, "expenses"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_expense_log (
              id TEXT PRIMARY KEY,
              amount DOUBLE PRECISION NOT NULL,
              currency TEXT NOT NULL DEFAULT '',
              category TEXT NOT NULL DEFAULT '',
              note TEXT NOT NULL DEFAULT '',
              spent_on TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "expenses")


def _entry_from_row(row: dict):
    from ..expenses.store import ExpenseEntry

    return ExpenseEntry(
        id=str(row["id"]),
        amount=float(row.get("amount") or 0.0),
        currency=str(row.get("currency") or ""),
        category=str(row.get("category") or ""),
        note=str(row.get("note") or ""),
        spent_on=str(row.get("spent_on") or ""),
        created=str(row.get("created") or ""),
    )


def log_expense_entry(
    settings: Settings,
    amount: float,
    currency: str = "",
    category: str = "",
    note: str = "",
    on: str = "",
):
    import uuid

    from ..expenses import store as expense_store

    ensure_expenses_schema(settings)
    spent_on = expense_store.parse_date(on) or expense_store._today(settings)
    entry = expense_store.ExpenseEntry(
        id=uuid.uuid4().hex[:12],
        amount=float(amount),
        currency=currency.strip(),
        category=category.strip().lower(),
        note=note.strip(),
        spent_on=spent_on.isoformat(),
        created=expense_store._stamp_now(settings),
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_expense_log"
            " (id, amount, currency, category, note, spent_on, created)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (entry.id, entry.amount, entry.currency, entry.category, entry.note,
             entry.spent_on, entry.created),
        )
    return entry


def ensure_budget_schema(settings: Settings) -> None:
    if _schema_done(settings, "expense_budgets"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_expense_budgets (
              category TEXT PRIMARY KEY,
              amount DOUBLE PRECISION NOT NULL,
              currency TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "expense_budgets")


def set_expense_budget(settings: Settings, budget):
    ensure_budget_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_expense_budgets (category, amount, currency)"
            " VALUES (%s, %s, %s)"
            " ON CONFLICT (category) DO UPDATE SET amount = EXCLUDED.amount,"
            " currency = EXCLUDED.currency",
            (budget.category, budget.amount, budget.currency),
        )
    return budget


def list_expense_budgets(settings: Settings):
    from ..expenses.store import Budget

    ensure_budget_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute("SELECT category, amount, currency FROM assistant_expense_budgets")
        )
    return [
        Budget(
            category=str(row.get("category") or ""),
            amount=float(row.get("amount") or 0.0),
            currency=str(row.get("currency") or ""),
        )
        for row in rows
    ]


def remove_expense_budget(settings: Settings, category: str):
    existing = next(
        (b for b in list_expense_budgets(settings) if b.category == category), None
    )
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute(
            "DELETE FROM assistant_expense_budgets WHERE category = %s", (category,)
        )
    return existing


def get_expense_entry(settings: Settings, entry_id: str):
    ensure_expenses_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(f"SELECT {_COLS} FROM assistant_expense_log WHERE id = %s", (entry_id,))
        )
    return _entry_from_row(rows[0]) if rows else None


def list_expense_entries(settings: Settings):
    ensure_expenses_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_expense_log"))
    return [_entry_from_row(r) for r in rows]


def delete_expense_entry(settings: Settings, entry_id: str):
    existing = get_expense_entry(settings, entry_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_expense_log WHERE id = %s", (entry_id,))
    return existing
