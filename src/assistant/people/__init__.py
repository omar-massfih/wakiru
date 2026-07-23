"""The people the user knows — a lightweight, offline CRM.

A self-contained subsystem mirroring :mod:`assistant.tasks`, but for the people
in the user's life rather than their to-dos:

* **Read** — :func:`people_context` injects a compact roster into every turn
  (wired via :mod:`assistant.context_providers`), with anyone overdue for
  contact or with a birthday coming flagged first, so "who is …?" enrichment and
  "reach out to X" prompting both work.
* **Write** — the agent's people tools (:mod:`assistant.tools`) add, update, log
  contact with, and remove people through :func:`.ops.apply_op`.

People live in a SQLite store (:mod:`.store`) under the memory directory, with a
parallel undo ledger (:mod:`.undo`) that the cross-subsystem "undo" arbiter
(:mod:`assistant.undo`) consults alongside the calendar's and the tasks'.
"""

from __future__ import annotations

from . import store
from .context import attention_lines, briefing_people, people_context
from .store import Person

__all__ = [
    "Person",
    "attention_lines",
    "briefing_people",
    "people_context",
    "store",
]
