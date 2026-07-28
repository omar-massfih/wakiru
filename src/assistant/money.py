"""Money/number formatting shared by the expense, habit, and subscription views."""

from __future__ import annotations


def num(value: float) -> str:
    """A number without a trailing ``.0`` (``12`` not ``12.0``), 2-dp otherwise."""
    return str(int(value)) if value == int(value) else str(round(value, 2))


def format_amount(amount: float, currency: str) -> str:
    """An amount with its currency appended (``99 kr``), trimming a whole ``.0``.

    An empty ``currency`` renders the bare number; callers that suppress a zero
    amount do so before calling.
    """
    suffix = f" {currency}" if currency else ""
    return f"{num(amount)}{suffix}"
