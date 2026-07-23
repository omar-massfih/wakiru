"""Trips — travel the assistant should be aware of, before and during.

A small, self-contained subsystem: a SQLite store (:mod:`.store`) under the
memory directory, the trip tools (:mod:`assistant.tools`), and a per-turn
context block (:func:`trips_context`) that surfaces the active trip (with the
destination's local time when a timezone is known) or the next departure while
it is imminent. Between trips it contributes nothing — no prompt bloat for the
eleven months a year nothing is booked.
"""

from __future__ import annotations

from . import store
from .context import trips_context
from .store import Trip

__all__ = ["Trip", "store", "trips_context"]
