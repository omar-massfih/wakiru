"""Watches — perception the model registers for itself.

The heartbeat's situation report used to observe a fixed set of things
(inbox hash, contact staleness); anything else meant another hardcoded
``_mail_changed``-style function. A watch turns that inside out: the model
says *what to look for* ("tell me when Skatteetaten writes", "wake me 30
minutes before the flight", "flag it if I haven't heard from the user by
Friday noon"), and the deterministic pre-check evaluates it every wake —
token-free — raising a trigger with the model's own note-to-self when it
fires.

Three kinds, fixed on purpose (substring patterns, no DSL):

* ``mail_from`` — the cached unread snapshot newly contains a line matching
  the pattern (sender or subject). May repeat.
* ``calendar_window`` — now entered ``[start - lead, start]`` of an upcoming
  event whose title matches the pattern. The scheduler also wakes for it.
* ``silence`` — no user message arrived by a deadline.

Watches are one-shot by default (claim-first status flip, the followups
discipline), carry a mandatory expiry so a stale watch cannot haunt the
report forever, and are capped so watch spam cannot crowd out the wake.
Rows live in ``followups.db``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import followups, threads
from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings, postgres_backend

logger = logging.getLogger(__name__)

KINDS = ("mail_from", "calendar_window", "silence")
DEFAULT_LEAD_MINUTES = 30
# A fired-or-expired silence watch lingers a day past its deadline so the
# evaluation (deadline) always precedes the expiry sweep.
_SILENCE_GRACE = timedelta(days=1)


@dataclass(frozen=True)
class Watch:
    id: str
    kind: str
    pattern: str
    note: str = ""
    lead_minutes: int = DEFAULT_LEAD_MINUTES
    repeat: bool = False
    fire_at: str = ""  # silence only: the deadline
    expires_at: str = ""
    last_match_hash: str = ""
    created_at: str = ""
    status: str = "active"  # active | fired | cancelled | expired


def _open(settings: Settings) -> sqlite3.Connection:
    conn = followups._open(settings)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS watches ("
        " id TEXT PRIMARY KEY, kind TEXT NOT NULL, pattern TEXT NOT NULL,"
        " note TEXT NOT NULL DEFAULT '',"
        f" lead_minutes INTEGER NOT NULL DEFAULT {DEFAULT_LEAD_MINUTES},"
        " repeat INTEGER NOT NULL DEFAULT 0, fire_at TEXT NOT NULL DEFAULT '',"
        " expires_at TEXT NOT NULL DEFAULT '', last_match_hash TEXT NOT NULL DEFAULT '',"
        " created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    from .sqlite_util import transaction

    with transaction(_open(settings)) as conn:
        yield conn


def _from_row(row: sqlite3.Row) -> Watch:
    data = dict(row)
    data["repeat"] = bool(data.get("repeat"))
    return Watch(**data)


def add(
    settings: Settings,
    kind: str,
    pattern: str,
    note: str = "",
    until: datetime | None = None,
    repeat: bool = False,
    lead_minutes: int = DEFAULT_LEAD_MINUTES,
) -> Watch | None:
    """Register one watch; ``None`` when the kind is unknown or the cap is hit.

    ``until`` is the expiry — mandatory in effect: omitted, it defaults to
    ``watch_default_expiry_days`` out. For ``silence`` it is the deadline
    itself (the watch then lingers a grace day so the deadline check runs
    before the expiry sweep).
    """
    kind = str(kind).strip()
    if kind not in KINDS:
        return None
    if len(list_active(settings)) >= max(settings.watches_max_active, 0):
        return None
    current = now(settings)
    expiry = until or current + timedelta(days=max(settings.watch_default_expiry_days, 1))
    fire_at = ""
    if kind == "silence":
        fire_at = expiry.isoformat(timespec="seconds")
        expiry = expiry + _SILENCE_GRACE
    watch = Watch(
        id=uuid.uuid4().hex[:12],
        kind=kind,
        pattern=str(pattern).strip(),
        note=str(note).strip(),
        lead_minutes=max(int(lead_minutes), 0),
        repeat=bool(repeat) and kind == "mail_from",
        fire_at=fire_at,
        expires_at=expiry.isoformat(timespec="seconds"),
        created_at=current.isoformat(timespec="seconds"),
    )
    if storage_postgres := postgres_backend(settings):
        storage_postgres.add_watch(settings, watch)
        return watch
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO watches"
            " (id, kind, pattern, note, lead_minutes, repeat, fire_at,"
            "  expires_at, last_match_hash, created_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?, 'active')",
            (
                watch.id,
                watch.kind,
                watch.pattern,
                watch.note,
                watch.lead_minutes,
                int(watch.repeat),
                watch.fire_at,
                watch.expires_at,
                watch.created_at,
            ),
        )
    return watch


def list_active(settings: Settings, current: datetime | None = None) -> list[Watch]:
    """Active watches, expiring the overdue ones on the way (mutes-style prune)."""
    current = current or now(settings)
    stamp = current.isoformat(timespec="seconds")
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.list_active_watches(settings, stamp, current)
    with _connect(settings) as conn:
        rows = conn.execute("SELECT * FROM watches WHERE status = 'active'").fetchall()
        expired = [
            row["id"]
            for row in rows
            if (until := parse_dt(row["expires_at"])) is not None and until <= current
        ]
        conn.executemany(
            "UPDATE watches SET status = 'expired' WHERE id = ? AND status = 'active'",
            [(wid,) for wid in expired],
        )
    dropped = set(expired)
    return [_from_row(row) for row in rows if row["id"] not in dropped]


def _match_active(settings: Settings, ident: str, action: str) -> Watch | None:
    """Resolve an active watch by id or an unambiguous pattern/note reference."""
    ident = str(ident).strip()
    if not ident:
        return None
    candidates = list_active(settings)
    matches = [w for w in candidates if w.id == ident]
    if not matches:
        needle = ident.lower()
        matches = [
            w
            for w in candidates
            if needle in w.pattern.lower() or needle in w.note.lower()
        ]
    if len(matches) != 1:
        if len(matches) > 1:
            logger.warning(
                "watch %s target %r is ambiguous between %d watches; skipping",
                action,
                ident,
                len(matches),
            )
        return None
    return matches[0]


def cancel(settings: Settings, ident: str) -> Watch | None:
    target = _match_active(settings, ident, "cancel")
    if target is None:
        return None
    if storage_postgres := postgres_backend(settings):
        return target if storage_postgres.cancel_watch(settings, target.id) else None
    with _connect(settings) as conn:
        cursor = conn.execute(
            "UPDATE watches SET status = 'cancelled' WHERE id = ? AND status = 'active'",
            (target.id,),
        )
    return target if cursor.rowcount else None


def _claim(settings: Settings, watch: Watch, match_hash: str = "") -> bool:
    """Consume a firing: repeat watches store the new match hash and stay
    active; one-shots flip to fired, claim-first (exactly-once)."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.claim_watch(
            settings, watch.id, watch.repeat, match_hash
        )
    with _connect(settings) as conn:
        if watch.repeat:
            cursor = conn.execute(
                "UPDATE watches SET last_match_hash = ?"
                " WHERE id = ? AND status = 'active' AND last_match_hash != ?",
                (match_hash, watch.id, match_hash),
            )
        else:
            cursor = conn.execute(
                "UPDATE watches SET status = 'fired', last_match_hash = ?"
                " WHERE id = ? AND status = 'active'",
                (match_hash, watch.id),
            )
    return bool(cursor.rowcount)


# --------------------------------------------------------------------------- #
# Evaluation — deterministic, token-free, run by every heartbeat pre-check
# --------------------------------------------------------------------------- #

def _mail_matches(settings: Settings, pattern: str) -> list[str]:
    if not (settings.enable_email and settings.email_snapshot_minutes > 0):
        return []
    from .mail.snapshot import current as snapshot_current

    needle = pattern.lower()
    return [
        line.strip()
        for line in snapshot_current(settings).splitlines()
        if needle and needle in line.lower()
    ]


def _matched_events(settings: Settings, pattern: str) -> list:
    from .calendar import upcoming_events

    needle = pattern.lower()
    return [e for e in upcoming_events(settings) if needle and needle in e.title.lower()]


def evaluate(settings: Settings, current: datetime | None = None) -> list[tuple[Watch, str]]:
    """Fired watches with their trigger lines; each one-shot exactly once."""
    current = current or now(settings)
    fired: list[tuple[Watch, str]] = []
    for watch in list_active(settings, current):
        try:
            line = _evaluate_one(settings, watch, current)
        except Exception:
            logger.exception("evaluating watch %s (%s) failed", watch.id, watch.kind)
            continue
        if line:
            fired.append((watch, line))
    return fired


def _evaluate_one(settings: Settings, watch: Watch, current: datetime) -> str:
    note = f" — your note: {watch.note}" if watch.note else ""
    if watch.kind == "mail_from":
        lines = _mail_matches(settings, watch.pattern)
        if not lines:
            return ""
        digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        if digest == watch.last_match_hash:
            return ""
        if not _claim(settings, watch, digest):
            return ""
        return (
            f'Watch hit — unread mail matching "{watch.pattern}": '
            + "; ".join(lines[:3])
            + note
        )
    if watch.kind == "calendar_window":
        for event in _matched_events(settings, watch.pattern):
            start = parse_dt(event.start)
            if start is None:
                continue
            lead = timedelta(minutes=watch.lead_minutes)
            if start - lead <= current <= start and _claim(settings, watch):
                return (
                    f'Watch hit — "{event.title}" starts at {event.start} '
                    f"(within your {watch.lead_minutes}-minute lead){note}"
                )
        return ""
    if watch.kind == "silence":
        deadline = parse_dt(watch.fire_at)
        if deadline is None or current < deadline:
            return ""
        last = threads.last_contact(settings)
        created = parse_dt(watch.created_at)
        heard = last is not None and (created is None or last >= created)
        if heard:
            # The user did write — the watch's condition can never fire now.
            _claim(settings, watch)
            return ""
        if not _claim(settings, watch):
            return ""
        return (
            f"Watch hit — the user has not written since you set this watch "
            f"and its deadline ({watch.fire_at}) passed{note}"
        )
    return ""


def wake_times(settings: Settings, current: datetime | None = None) -> list[datetime]:
    """Times the scheduler should wake for: calendar-window opens and silence
    deadlines. Mail watches ride the snapshot cadence instead."""
    current = current or now(settings)
    times: list[datetime] = []
    for watch in list_active(settings, current):
        try:
            if watch.kind == "silence":
                deadline = parse_dt(watch.fire_at)
                if deadline is not None and deadline > current:
                    times.append(deadline)
            elif watch.kind == "calendar_window":
                for event in _matched_events(settings, watch.pattern):
                    start = parse_dt(event.start)
                    if start is None:
                        continue
                    opens = start - timedelta(minutes=watch.lead_minutes)
                    if opens > current:
                        times.append(opens)
        except Exception:
            logger.exception("computing wake time for watch %s failed", watch.id)
    return times
