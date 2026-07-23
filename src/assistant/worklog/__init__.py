"""Chat-driven work log — where the user's working time actually went.

A self-contained subsystem: an append-only SQLite log (:mod:`.store`) of
stretches of work on named projects — a live start/stop timer plus
after-the-fact logs — and the worklog tools (:mod:`assistant.tools`), which
drive it and roll days or weeks up per project (:mod:`.context`). The work
twin of :mod:`assistant.expenses`: that answers "where did the money go",
this answers "where did the time go" — and the weekly review carries last
week's hours per project.
"""

from __future__ import annotations

from . import context, store
from .store import WorkEntry

__all__ = ["WorkEntry", "context", "store"]
