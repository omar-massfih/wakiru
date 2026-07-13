"""Shared fired-ledger tests — claim-once semantics and retention pruning.

The subsystem behaviors riding on this driver (reminder dedupe, briefing
once-per-day) are covered in test_reminders.py / test_briefing.py; here the
driver itself is exercised against a scratch spec.
"""

from __future__ import annotations

from datetime import UTC, timedelta

import pytest

from assistant import fired_ledger
from assistant.calendar.context import now
from assistant.config import Settings
from assistant.fired_ledger import FiredLedgerSpec

_SPEC = FiredLedgerSpec(
    table="things_fired",
    columns=(("thing_id", "TEXT"), ("slot", "INTEGER")),
    db_path=lambda settings: settings.memory_path / "things.db",
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), timezone="Europe/Oslo")


def _rows(settings: Settings) -> list[dict]:
    with fired_ledger._connect(_SPEC, settings) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM things_fired").fetchall()]


def test_claim_returns_indexes_of_new_keys_only(settings) -> None:
    current = now(settings)
    fired_at = current.isoformat(timespec="seconds")

    first = fired_ledger.claim(_SPEC, settings, [("a", 1), ("b", 1)], fired_at, current)
    assert first == [0, 1]

    # A second pass with one old and one new key claims only the new one.
    second = fired_ledger.claim(_SPEC, settings, [("a", 1), ("c", 1)], fired_at, current)
    assert second == [1]
    assert len(_rows(settings)) == 3


def test_claim_is_idempotent(settings) -> None:
    current = now(settings)
    fired_at = current.isoformat(timespec="seconds")
    keys = [("a", 1)]
    assert fired_ledger.claim(_SPEC, settings, keys, fired_at, current) == [0]
    assert fired_ledger.claim(_SPEC, settings, keys, fired_at, current) == []


def test_prune_drops_old_and_unparseable_rows(settings) -> None:
    current = now(settings)
    old = (current - timedelta(days=40)).isoformat(timespec="seconds")
    fresh = (current - timedelta(days=1)).isoformat(timespec="seconds")
    with fired_ledger._connect(_SPEC, settings) as conn:
        conn.executemany(
            "INSERT INTO things_fired (thing_id, slot, fired_at) VALUES (?, ?, ?)",
            [("stale", 1, old), ("garbled", 1, "not-a-date"), ("fresh", 1, fresh)],
        )

    fired_ledger.claim(_SPEC, settings, [], current.isoformat(timespec="seconds"), current)
    assert [r["thing_id"] for r in _rows(settings)] == ["fresh"]


def test_prune_compares_instants_not_strings(settings) -> None:
    # A fresh row stamped under another UTC offset sorts lexically before the
    # cutoff string; pruning must compare instants and keep it.
    current = now(settings)
    other_offset = (current - timedelta(days=1)).astimezone(UTC)
    with fired_ledger._connect(_SPEC, settings) as conn:
        conn.execute(
            "INSERT INTO things_fired (thing_id, slot, fired_at) VALUES (?, ?, ?)",
            ("fresh", 1, other_offset.isoformat(timespec="seconds")),
        )
    fired_ledger.claim(_SPEC, settings, [], current.isoformat(timespec="seconds"), current)
    assert [r["thing_id"] for r in _rows(settings)] == ["fresh"]


def test_prune_takes_naive_stamps_as_utc(settings) -> None:
    # The pre-unification briefing ledger wrote naive UTC stamps via SQLite's
    # datetime('now'); a fresh naive row must survive the prune.
    current = now(settings)
    naive_utc = (
        (current - timedelta(days=1))
        .astimezone(UTC)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%d %H:%M:%S")
    )
    with fired_ledger._connect(_SPEC, settings) as conn:
        conn.execute(
            "INSERT INTO things_fired (thing_id, slot, fired_at) VALUES (?, ?, ?)",
            ("legacy", 1, naive_utc),
        )
    fired_ledger.claim(_SPEC, settings, [], current.isoformat(timespec="seconds"), current)
    assert [r["thing_id"] for r in _rows(settings)] == ["legacy"]
