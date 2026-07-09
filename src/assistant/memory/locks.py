"""Process-wide serialization of long-term-memory mutations.

The memory store is a set of markdown files plus a derived SQLite index, mutated
from several threads at once: FastAPI background upkeep, the Telegram channel's
upkeep tasks, turn-triggered and on-demand consolidation, and the startup
reindex. None of those writes are atomic across the two layers, so without
coordination they can lose a note (two saves picking the same free name),
resurrect a forgotten one (revise racing a delete), or leave recall blind while
``reindex`` has the vector table dropped.

One re-entrant lock closes all of those windows. It guards *short* file+index
critical sections only — never an LLM call, which can run for minutes: the
learners take the lock per applied op, not around the whole extraction.
:data:`CONSOLIDATE_LOCK` separately keeps whole consolidation passes mutually
exclusive (a pass overlapping itself just wastes Codex tokens re-reading the
same episodes) without holding up replies while consolidation thinks.
"""

from __future__ import annotations

import threading
from functools import wraps

MEMORY_LOCK = threading.RLock()

# Non-blocking guard for consolidation passes; see consolidate_memory.
CONSOLIDATE_LOCK = threading.Lock()


def locked(fn):
    """Run ``fn`` while holding :data:`MEMORY_LOCK` (re-entrant, so locked
    functions may call each other)."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with MEMORY_LOCK:
            return fn(*args, **kwargs)

    return wrapper
