"""SQLite store for the expense log.

A single ``expense_log`` table in its own SQLite file
(:attr:`Settings.expenses_db_path`). Each row is one logged expense — an
``amount`` with an optional ``currency``, an optional ``category`` + ``note``,
and the ``spent_on`` date. Like the habits log this is an append log, not a
mutable record set; the read path (:mod:`.context`) rolls a month up by
currency and category over it.

A fresh connection is opened per operation with WAL + a busy timeout, so the
store is safe from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime

from ..config import Settings, postgres_backend
from ..sqlite_util import open_db, transaction


@dataclass
class ExpenseEntry:
    """One logged expense.

    ``amount`` is always positive. ``currency`` is free text ("kr", "EUR") —
    rollups group by it, an empty currency groups under "?". ``spent_on`` is a
    ``YYYY-MM-DD`` date; ``created`` is the tz-aware ISO stamp, used to order
    entries within a day.
    """

    id: str
    amount: float
    currency: str = ""
    category: str = ""
    note: str = ""
    spent_on: str = ""
    created: str = ""


def parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_month(value: str) -> str:
    """Normalize a ``YYYY-MM`` month string; empty when unparseable."""
    value = (value or "").strip()[:7]
    try:
        datetime.strptime(value, "%Y-%m")
    except ValueError:
        return ""
    return value


def _coerce_amount(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return default


def _today(settings: Settings) -> date:
    from ..calendar.context import now

    return now(settings).date()


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.expenses_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS expense_log ("
        " id TEXT PRIMARY KEY, amount REAL NOT NULL,"
        " currency TEXT DEFAULT '', category TEXT DEFAULT '',"
        " note TEXT DEFAULT '', spent_on TEXT DEFAULT '', created TEXT DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def _row_to_entry(row: sqlite3.Row) -> ExpenseEntry:
    return ExpenseEntry(
        id=row["id"],
        amount=float(row["amount"] or 0.0),
        currency=row["currency"] or "",
        category=row["category"] or "",
        note=row["note"] or "",
        spent_on=row["spent_on"] or "",
        created=row["created"] or "",
    )


def log_entry(
    settings: Settings,
    amount: object,
    currency: str = "",
    category: str = "",
    note: str = "",
    on: str = "",
) -> ExpenseEntry | None:
    """Append one expense and return it; ``None`` when the amount isn't positive.

    ``on`` defaults to today.
    """
    value = _coerce_amount(amount)
    if value <= 0:
        return None
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.log_expense_entry(settings, value, currency, category, note, on)
    spent_on = parse_date(on) or _today(settings)
    entry = ExpenseEntry(
        id=uuid.uuid4().hex[:12],
        amount=value,
        currency=currency.strip(),
        category=category.strip().lower(),
        note=note.strip(),
        spent_on=spent_on.isoformat(),
        created=_stamp_now(settings),
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO expense_log (id, amount, currency, category, note, spent_on, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry.id, entry.amount, entry.currency, entry.category, entry.note,
             entry.spent_on, entry.created),
        )
    return entry


def get_entry(settings: Settings, entry_id: str) -> ExpenseEntry | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_expense_entry(settings, entry_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM expense_log WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_entry(row) if row else None


def list_entries(settings: Settings, month: str = "", category: str = "") -> list[ExpenseEntry]:
    """Entries newest-first (by date, then created). ``month`` filters to a
    ``YYYY-MM``; ``category`` filters by name (case-insensitive exact match)."""
    if storage_postgres := postgres_backend(settings):
        entries = storage_postgres.list_expense_entries(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM expense_log").fetchall()
        entries = [_row_to_entry(r) for r in rows]
    if prefix := parse_month(month):
        entries = [e for e in entries if e.spent_on.startswith(prefix)]
    needle = category.strip().lower()
    if needle:
        entries = [e for e in entries if e.category == needle]
    return sorted(entries, key=lambda e: (e.spent_on, e.created), reverse=True)


def delete_entry(settings: Settings, entry_id: str) -> ExpenseEntry | None:
    """Delete one logged expense by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_expense_entry(settings, entry_id)
    existing = get_entry(settings, entry_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM expense_log WHERE id = ?", (entry_id,))
    return existing


def category_names(settings: Settings) -> list[str]:
    """Distinct categories, most-recently-used first (uncategorized excluded)."""
    seen: dict[str, None] = {}
    for entry in list_entries(settings):  # already newest-first
        if entry.category:
            seen.setdefault(entry.category, None)
    return list(seen)
