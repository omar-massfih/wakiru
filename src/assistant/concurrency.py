"""Shared scaffolding for the subprocess/HTTP backends.

The Codex CLI runner, the chatgpt.com backend, and the code sandbox each bound
their concurrent work with the same lazily-sized semaphore, and the two LLM
backends raise the same error/timeout pair. Those shared shapes live here so
each backend keeps only what is genuinely its own (argv/payload building,
stream parsing, auth).
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from .config import Settings


class BackendError(RuntimeError):
    """An LLM backend call (the Codex CLI or the chatgpt.com endpoint) failed."""


class BackendTimeoutError(BackendError):
    """A backend call exceeded its wall-clock / socket timeout.

    A subclass so ``except BackendError`` callers keep working, while channels
    can tell "took too long" from "broke" when explaining a failure.
    """


class BoundedSlot:
    """A lazily-created, process-wide concurrency semaphore sized from settings.

    Each backend caps how much runs at once with one ``BoundedSemaphore`` sized
    from a settings max (floor 1), created on the first call and reused after —
    settings are effectively a singleton, so one slot per backend is enough. A
    module binds one instance at import and calls it per operation
    (``with slot(settings): ...``); :meth:`reset` drops the cached semaphore so
    the next call re-sizes it (used by tests).
    """

    def __init__(self, get_max: Callable[[Settings], int]) -> None:
        self._get_max = get_max
        self._lock = threading.Lock()
        self._semaphore: threading.BoundedSemaphore | None = None

    def __call__(self, settings: Settings) -> threading.BoundedSemaphore:
        with self._lock:
            if self._semaphore is None:
                self._semaphore = threading.BoundedSemaphore(max(self._get_max(settings), 1))
            return self._semaphore

    def reset(self) -> None:
        with self._lock:
            self._semaphore = None
