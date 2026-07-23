"""The read-it-later list — links the user wants to get back to.

A small, self-contained subsystem: a SQLite store (:mod:`.store`) under the
memory directory and the reading tools (:mod:`assistant.tools`), which save,
list, mark-read, and remove saved links through it. Deliberately *not* injected
into every turn (it would bloat the prompt); the model reaches it with
``list_reading`` when the user asks "what's on my reading list?".
"""

from __future__ import annotations

from . import store
from .store import ReadingItem

__all__ = ["ReadingItem", "store"]
