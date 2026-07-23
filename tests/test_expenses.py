"""Expense-log tests — store append/list/filter, the monthly rollup, the
briefing's start-of-month section, and the tool paths.
"""

from __future__ import annotations

from datetime import date

import pytest

from assistant.config import Settings
from assistant.expenses import store
from assistant.expenses.context import (
    briefing_expenses,
    budget_lines,
    month_summary,
    totals_by_category,
    totals_by_currency,
)
from assistant.tools import ToolContext, tool_map


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_expenses=True,
    )


# --- store ------------------------------------------------------------------ #


def test_log_and_list(settings) -> None:
    entry = store.log_entry(settings, "250", currency="kr", category="Groceries")
    assert entry is not None and entry.amount == 250.0
    assert entry.category == "groceries"  # normalized lower-case
    entries = store.list_entries(settings)
    assert len(entries) == 1 and entries[0].currency == "kr"


def test_non_positive_amount_refused(settings) -> None:
    assert store.log_entry(settings, "0") is None
    assert store.log_entry(settings, "nonsense") is None
    assert store.list_entries(settings) == []


def test_list_filters_by_month_and_category(settings) -> None:
    store.log_entry(settings, "100", category="food", on="2026-06-15")
    store.log_entry(settings, "200", category="food", on="2026-07-01")
    store.log_entry(settings, "300", category="transport", on="2026-07-02")
    assert len(store.list_entries(settings, month="2026-07")) == 2
    assert [e.amount for e in store.list_entries(settings, month="2026-07", category="food")] == [200.0]
    assert store.list_entries(settings, month="not-a-month") == store.list_entries(settings)


def test_totals_group_by_currency_and_category(settings) -> None:
    store.log_entry(settings, "100", currency="kr", category="food", on="2026-07-01")
    store.log_entry(settings, "50", currency="kr", category="food", on="2026-07-02")
    store.log_entry(settings, "20", currency="EUR", category="travel", on="2026-07-03")
    entries = store.list_entries(settings, month="2026-07")
    assert totals_by_currency(entries) == {"kr": 150.0, "EUR": 20.0}
    cats = totals_by_category(entries)
    assert cats["food"] == {"kr": 150.0}
    assert list(cats) == ["food", "travel"]  # largest first


def test_month_summary_renders(settings) -> None:
    store.log_entry(settings, "250", currency="kr", category="groceries",
                    note="Rema", on="2026-07-05")
    text = month_summary(settings, "2026-07", date(2026, 7, 23))
    assert "2026-07" in text and "groceries" in text and "id:" in text
    assert "month to date" in text
    assert "No expenses" in month_summary(settings, "2026-01", date(2026, 7, 23))


# --- budgets ------------------------------------------------------------------ #


def test_set_list_and_remove_budget(settings) -> None:
    assert store.set_budget(settings, "Groceries", "3000", currency="kr") is not None
    assert store.set_budget(settings, "total", "10000", currency="kr") is not None
    budgets = store.list_budgets(settings)
    assert [b.category for b in budgets] == ["", "groceries"]  # overall first
    store.set_budget(settings, "groceries", "3500", currency="kr")  # upsert
    assert [b.amount for b in store.list_budgets(settings)] == [10000.0, 3500.0]
    removed = store.remove_budget(settings, "overall")
    assert removed is not None and removed.category == ""
    assert store.remove_budget(settings, "nope") is None
    assert len(store.list_budgets(settings)) == 1


def test_budget_non_positive_amount_refused(settings) -> None:
    assert store.set_budget(settings, "food", "0") is None
    assert store.set_budget(settings, "food", "nonsense") is None
    assert store.list_budgets(settings) == []


def test_budget_lines_track_spend_and_overrun(settings) -> None:
    store.set_budget(settings, "groceries", "1000", currency="kr")
    store.log_entry(settings, "800", currency="kr", category="groceries", on="2026-07-05")
    store.log_entry(settings, "20", currency="EUR", category="groceries", on="2026-07-06")
    lines = budget_lines(settings, "2026-07")
    assert lines == ["groceries: 800 of 1000 kr (80%)"]  # EUR entry doesn't count
    store.log_entry(settings, "400", currency="kr", category="groceries", on="2026-07-10")
    assert budget_lines(settings, "2026-07") == [
        "groceries: 1200 of 1000 kr (120%) — over by 200 kr"
    ]
    assert budget_lines(settings, "2026-06") == ["groceries: 0 of 1000 kr (0%)"]


def test_overall_budget_without_currency_counts_everything(settings) -> None:
    store.set_budget(settings, "", "1000")
    store.log_entry(settings, "300", currency="kr", category="food", on="2026-07-01")
    store.log_entry(settings, "200", currency="EUR", category="travel", on="2026-07-02")
    assert budget_lines(settings, "2026-07") == ["overall: 500 of 1000 (50%)"]


def test_month_summary_includes_budget_status(settings) -> None:
    store.set_budget(settings, "groceries", "1000", currency="kr")
    store.log_entry(settings, "250", currency="kr", category="groceries", on="2026-07-05")
    text = month_summary(settings, "2026-07", date(2026, 7, 23))
    assert "budgets:" in text and "250 of 1000 kr (25%)" in text


# --- briefing section ------------------------------------------------------- #


def test_briefing_rolls_up_last_month_on_the_first(settings) -> None:
    store.log_entry(settings, "500", currency="kr", category="food", on="2026-06-20")
    block = briefing_expenses(settings, date(2026, 7, 1))
    assert "2026-06" in block and "500 kr" in block and "food" in block
    assert "id:" not in block


def test_briefing_compares_last_month_against_budgets(settings) -> None:
    store.set_budget(settings, "food", "400", currency="kr")
    store.log_entry(settings, "500", currency="kr", category="food", on="2026-06-20")
    block = briefing_expenses(settings, date(2026, 7, 1))
    assert "Against budgets:" in block
    assert "food: 500 of 400 kr (125%) — over by 100 kr" in block


def test_briefing_empty_off_the_first_or_without_entries(settings) -> None:
    store.log_entry(settings, "500", on="2026-06-20")
    assert briefing_expenses(settings, date(2026, 7, 2)) == ""
    assert briefing_expenses(settings, date(2026, 8, 1)) == ""  # July has none


# --- tools ------------------------------------------------------------------ #


def _run(settings, tool, **args) -> str:
    return tool_map(settings)[tool].run(ToolContext(settings=settings), **args)


def test_log_expense_tool(settings) -> None:
    out = _run(settings, "log_expense", amount="49.90", currency="kr", category="coffee")
    assert "Logged 49.9 kr on coffee" in out
    assert len(store.list_entries(settings)) == 1
    assert "positive amount" in _run(settings, "log_expense", amount="-5")


def test_expense_summary_and_remove_tools(settings) -> None:
    _run(settings, "log_expense", amount="120", currency="kr", category="lunch")
    summary = _run(settings, "expense_summary")
    assert "lunch" in summary and "120 kr" in summary
    entry = store.list_entries(settings)[0]
    assert "Removed" in _run(settings, "remove_expense", id=entry.id)
    assert store.list_entries(settings) == []
    assert "No expense with id" in _run(settings, "remove_expense", id="nope")


def test_budget_tools(settings) -> None:
    out = _run(settings, "set_budget", amount="3000", category="Groceries", currency="kr")
    assert "Budget set: groceries — 3000 kr per month." in out
    assert "positive amount" in _run(settings, "set_budget", amount="-5")
    out = _run(settings, "set_budget", amount="10000", currency="kr")
    assert "overall spending" in out
    summary = _run(settings, "expense_summary")
    assert "budgets:" in summary and "groceries" in summary and "overall" in summary
    assert "Removed the groceries budget." in _run(settings, "remove_budget", category="groceries")
    assert "No groceries budget is set." in _run(settings, "remove_budget", category="groceries")
    assert "Removed the overall budget." in _run(settings, "remove_budget")


def test_tools_absent_when_disabled(settings) -> None:
    off = settings.model_copy(update={"enable_expenses": False})
    assert "log_expense" not in tool_map(off)
    assert "set_budget" not in tool_map(off)


def test_writes_chat_only_summary_stays_in_heartbeat(settings) -> None:
    from assistant.tools import available_tools

    names = {spec.name for spec in available_tools(settings, mode="heartbeat")}
    assert "log_expense" not in names and "remove_expense" not in names
    assert "set_budget" not in names and "remove_budget" not in names
    assert "expense_summary" in names
