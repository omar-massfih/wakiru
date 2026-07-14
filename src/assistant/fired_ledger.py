"""Shared exactly-once "fired" ledger — one driver for every proactive push.

Calendar reminders, task reminders, and the daily briefing each need the same
guarantee: a push claimed once is never pushed again, even when the in-process
ticker and a manual endpoint race. Each used to carry its own copy of the
open/prune/claim SQLite boilerplate; this module is the single driver, the
same way :mod:`assistant.write_ledger` is for the undo ledgers.

A subsystem describes its ledger with a :class:`FiredLedgerSpec` (table name,
key columns, which DB file it lives in — all byte-identical to the tables the
copies created, so existing data keeps working) and claims keys through
:func:`claim`. Claiming is atomic per key (``INSERT OR IGNORE`` inside one
transaction) and each call prunes rows older than the retention window, so the
ledger stays bounded without a maintenance job.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

# Fired rows older than this are pruned on each claim so the ledger, which only
# ever grows, stays bounded.
LEDGER_RETENTION_DAYS = 30


@dataclass(frozen=True)
class FiredLedgerSpec:
    """Everything subsystem-specific about one fired ledger.

    ``columns`` are the key columns as (name, SQL type) pairs; together they form
    the primary key. A ``fired_at TEXT`` column is always appended.
    """

    table: str
    columns: tuple[tuple[str, str], ...]
    db_path: Callable[[Settings], Path]


def _open(spec: FiredLedgerSpec, settings: Settings) -> sqlite3.Connection:
    """Open the spec's DB and ensure its ledger table exists (WAL + busy timeout)."""
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(spec.db_path(settings))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    cols = ", ".join(f"{name} {sql_type} NOT NULL" for name, sql_type in spec.columns)
    key = ", ".join(name for name, _ in spec.columns)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {spec.table}"
        f" ({cols}, fired_at TEXT NOT NULL, PRIMARY KEY ({key}))"
    )
    return conn


@contextmanager
def connect(spec: FiredLedgerSpec, settings: Settings) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit."""
    conn = _open(spec, settings)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _parse_stamp(raw: str) -> datetime | None:
    """Parse a ``fired_at`` stamp; a naive stamp is taken as UTC.

    Rows written by the pre-unification briefing ledger used SQLite's
    ``datetime('now')`` (naive UTC, space-separated); everything else writes
    tz-aware ISO-8601. Both must prune correctly.
    """
    try:
        stamp = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _prune(spec: FiredLedgerSpec, conn: sqlite3.Connection, current: datetime) -> None:
    """Drop fired rows older than the retention window (and unparseable ones).

    Compared as datetimes in Python rather than as ISO strings in SQL: stamps
    written under different UTC offsets (a DST change) don't order lexically.
    The ledger holds at most ~30 days of rows, so the full scan is cheap.
    """
    cutoff = current - timedelta(days=LEDGER_RETENTION_DAYS)
    names = [name for name, _ in spec.columns]
    stale = [
        tuple(row[name] for name in names)
        for row in conn.execute(f"SELECT {', '.join(names)}, fired_at FROM {spec.table}")
        if (fired := _parse_stamp(row["fired_at"])) is None or fired < cutoff
    ]
    where = " AND ".join(f"{name} = ?" for name in names)
    conn.executemany(f"DELETE FROM {spec.table} WHERE {where}", stale)


def claim(
    spec: FiredLedgerSpec,
    settings: Settings,
    keys: Sequence[tuple],
    fired_at: str,
    current: datetime,
) -> list[int]:
    """Atomically claim each key, returning the indexes of keys newly claimed.

    A key already present (claimed by an earlier tick or a racing manual call)
    is silently skipped. All claims and the retention prune commit in one
    transaction; deliver only *after* it returns, so network I/O never holds
    SQLite's single writer slot (the claim-first / deliver-after discipline —
    at-most-once delivery, never a duplicate).
    """
    names = ", ".join(name for name, _ in spec.columns)
    slots = ", ".join("?" for _ in spec.columns)
    newly: list[int] = []
    with connect(spec, settings) as conn:
        _prune(spec, conn, current)
        for index, key in enumerate(keys):
            cursor = conn.execute(
                f"INSERT OR IGNORE INTO {spec.table} ({names}, fired_at)"
                f" VALUES ({slots}, ?)",
                (*key, fired_at),
            )
            if cursor.rowcount:
                newly.append(index)
    return newly


def fire_due(
    spec: FiredLedgerSpec,
    settings: Settings,
    agent: Any,
    due: list[dict],
    *,
    current: datetime,
    kind: str,
    key_fields: tuple[str, str],
    pg_claim: str,
    instruction: str,
    fact_line: Callable[[dict], str],
    deliver: Callable[[Settings, dict], bool],
    log_label: str,
) -> list[dict]:
    """Claim every ``due`` reminder exactly once and push the claimed batch.

    The pipeline calendar and task reminders share: quiet-hours hold → mute
    filter → exactly-once claim → one composed push → working-memory record.
    Each due dict carries ``title``, ``message``, ``covered_leads``, and the two
    ``key_fields`` (its id and its instant) that key the dedupe ledger together
    with each covered lead. ``pg_claim`` names the :mod:`assistant.storage_postgres`
    adapter, resolved at call time so test monkeypatches on that module keep
    working; ``deliver`` is passed in so callers keep their patchable module
    attribute. Best-effort and idempotent, so an in-process ticker and a manual
    trigger can both drive it safely. Quiet hours are the callers' job — they
    must hold *before* the due list is computed.
    """
    # Honor active mutes (the agent's "stop nudging me about this" switch):
    # filtered before the claim, so nudges resume on the first tick after a
    # mute expires. Deferred import: this low-level ledger must not pull the
    # compose/delivery stack (and its package cycles) in at import time.
    from .mutes import filter_muted

    due = filter_muted(settings, due, current, kind)
    fired_at = current.isoformat(timespec="seconds")

    # Claim first, commit, deliver after: delivery is network I/O (webhook POST,
    # a Telegram send per chat) and must not run inside the ledger's write
    # transaction, where it would hold SQLite's single writer slot past other
    # writers' busy timeouts. The cost is at-most-once delivery: a claimed
    # reminder whose push fails is not retried.
    if settings.storage_backend == "postgres":
        from . import storage_postgres

        sent = getattr(storage_postgres, pg_claim)(settings, due, fired_at, current)
    else:
        # Claim every lead window each reminder is currently inside, so the
        # larger leads can't fire a duplicate nudge on a later tick; a reminder
        # is sent when any of its windows was newly claimed.
        keys = [
            (*(reminder[field] for field in key_fields), lead)
            for reminder in due
            for lead in reminder["covered_leads"]
        ]
        owner = [
            index
            for index, reminder in enumerate(due)
            for _ in reminder["covered_leads"]
        ]
        claimed = claim(spec, settings, keys, fired_at, current)
        sent = [due[index] for index in sorted({owner[key_index] for key_index in claimed})]

    if not sent:
        return sent

    # One push per batch, composed by the model in the assistant's own voice
    # (memory and agenda ride in); the deterministic template text every
    # reminder already carries is the fallback, so a model failure degrades to
    # exactly the old behavior and never loses the nudge.
    from .compose import compose_push
    from .proactive import record_push

    text = compose_push(
        settings,
        instruction=instruction,
        facts="\n".join(fact_line(reminder) for reminder in sent),
        query=" ".join(reminder["title"] for reminder in sent),
        fallback="\n".join(reminder["message"] for reminder in sent),
    )
    try:
        delivered = deliver(settings, {"title": "Reminder", "message": text})
    except Exception:
        # The claim is already committed; delivery is best-effort by design.
        logger.exception("%s delivery failed: %s", log_label, text)
        return sent
    if delivered:
        # Recorded with the same ⏰ prefix the chat channels show, so the
        # thread's history matches what the user actually saw.
        record_push(agent, settings, f"⏰ {text}")

    logger.info("fired %d %s(s): %s", len(sent), log_label, text)
    return sent
