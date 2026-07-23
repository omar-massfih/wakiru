"""Chat-driven expense log — where the user's money actually went.

A self-contained subsystem: an append-only SQLite log (:mod:`.store`) of
one-off expenses (an amount + currency, a category, a note, a date) and the
expense tools (:mod:`assistant.tools`), which log entries and roll a month up
by currency and category (:mod:`.context`). The complement of
:mod:`assistant.subscriptions`: that tracks what *recurs*, this records what
was *spent* — "250 kr on groceries" — and the briefing opens each month with
last month's rollup.
"""

from __future__ import annotations

from . import context, store
from .store import ExpenseEntry

__all__ = ["ExpenseEntry", "context", "store"]
