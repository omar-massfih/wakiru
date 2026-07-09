"""Tests for :mod:`assistant.memory.locks`.

The lock coordinates memory mutations across threads. These verify the two
properties the rest of the system relies on: ``MEMORY_LOCK`` is re-entrant (so
``@locked`` functions may call one another without deadlocking), and it actually
serializes concurrent access.
"""

from __future__ import annotations

import threading
import time

from assistant.memory import locks


def test_memory_lock_is_reentrant() -> None:
    @locks.locked
    def inner() -> str:
        return "inner"

    @locks.locked
    def outer() -> str:
        # Re-acquiring MEMORY_LOCK from within a locked call must not deadlock.
        return inner() + "+outer"

    assert outer() == "inner+outer"


def test_locked_serializes_concurrent_access() -> None:
    order: list[str] = []

    @locks.locked
    def critical(tag: str) -> None:
        order.append(f"{tag}-enter")
        time.sleep(0.02)
        order.append(f"{tag}-exit")

    threads = [threading.Thread(target=critical, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Whoever entered first must exit before the other enters — no interleave.
    assert order[0].endswith("-enter")
    assert order[1].endswith("-exit")
    assert order[0].split("-")[0] == order[1].split("-")[0]


def test_consolidate_lock_is_non_blocking_guard() -> None:
    # A second acquire attempt while held fails immediately (used to skip an
    # overlapping consolidation pass rather than queue it).
    assert locks.CONSOLIDATE_LOCK.acquire(blocking=False) is True
    try:
        assert locks.CONSOLIDATE_LOCK.acquire(blocking=False) is False
    finally:
        locks.CONSOLIDATE_LOCK.release()
