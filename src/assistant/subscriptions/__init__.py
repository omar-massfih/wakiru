"""Recurring subscriptions / bills — track what the user pays and when it renews.

A self-contained subsystem mirroring :mod:`assistant.tasks`: a SQLite store
(:mod:`.store`), the subscription tools (:mod:`assistant.tools`), and proactive
renewal reminders (:mod:`.reminders`) that fire before a charge hits. The next
renewal is computed from a reference date + cadence, so a subscription's
schedule needs no upkeep. Low-stakes writes, so no undo ledger.
"""

from __future__ import annotations

from . import store
from .reminders import due_renewal_reminders, run_subscription_reminders
from .store import Subscription

__all__ = [
    "Subscription",
    "due_renewal_reminders",
    "run_subscription_reminders",
    "store",
]
