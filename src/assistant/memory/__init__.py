"""The assistant's memory (the "brain").

Two layers:

* **Working memory** — conversation history, via a LangGraph SQLite checkpointer
  wired in :mod:`assistant.agent` (keyed by ``thread_id``).
* **Long-term memory** — durable notes on disk (:mod:`.store`) with a local
  vector index (:mod:`.index`) for semantic recall (:mod:`.recall`) and both
  automatic and explicit learning (:mod:`.learn`).
"""

from __future__ import annotations

from .learn import forget_memory, save_memory, update_memory
from .recall import build_context_message, recall_context, search_memory
from .store import Note

__all__ = [
    "Note",
    "forget_memory",
    "save_memory",
    "update_memory",
    "build_context_message",
    "recall_context",
    "search_memory",
]
