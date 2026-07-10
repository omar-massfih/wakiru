"""The assistant's memory (the "brain").

Two layers:

* **Working memory** — conversation history, via a LangGraph SQLite checkpointer
  wired in :mod:`assistant.agent` (keyed by ``thread_id``), bounded by a rolling
  summary.
* **Long-term memory** — durable notes on disk (:mod:`.store`) in three kinds
  (episodic / semantic / procedural) with a local vector index (:mod:`.index`)
  for reinforcement-aware semantic recall (:mod:`.recall`), a reconciling learner
  (:mod:`.learn`), and a periodic consolidation pass (:mod:`.consolidate`).
"""

from __future__ import annotations

from . import index, store
from .consolidate import consolidate_memory
from .learn import forget_memory, record_episode, revise_memory, save_memory, update_memory
from .recall import build_context_message, recall_context, search_memory
from .store import Note

__all__ = [
    "Note",
    "build_context_message",
    "consolidate_memory",
    "forget_memory",
    "index",
    "recall_context",
    "record_episode",
    "revise_memory",
    "save_memory",
    "search_memory",
    "store",
    "update_memory",
]
