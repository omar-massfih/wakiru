"""Named checklists — shopping, errands, packing — distinct from dated tasks.

A small, self-contained subsystem: a SQLite store (:mod:`.store`) under the
memory directory and the list tools (:mod:`assistant.tools`), which add, show,
check off, and remove items through it. A list is just a name items share —
there is no separate lists table, and a list with no items does not exist.
Deliberately *not* injected into every turn (it would bloat the prompt); the
model reaches it with ``show_list`` when the user asks.
"""

from __future__ import annotations

from . import store
from .store import ListEntry

__all__ = ["ListEntry", "store"]
