"""SQLite store for recurring subscriptions / bills.

A single ``subscriptions`` table in its own SQLite file
(:attr:`Settings.subscriptions_db_path`). A subscription has a ``name``, an
``amount`` + ``currency``, a ``cadence`` (how often it renews), and a
``renews_on`` reference date. The next renewal is *computed* from that anchor and
cadence (like a birthday's next occurrence), so nothing mutates when a renewal
reminder fires — the fired ledger keys on the occurrence date for exactly-once.

A fresh connection is opened per operation with WAL + a busy timeout, so the
store is safe from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from ..config import Settings, postgres_backend
from ..sqlite_util import open_db, transaction

# Text columns a caller may set on update (amount is numeric, handled separately).
_TEXT_FIELDS = ("name", "currency", "cadence", "renews_on", "notes")

# Supported renewal cadences and their calendar step / months-per-cycle.
_DELTAS = {
    "weekly": relativedelta(weeks=1),
    "monthly": relativedelta(months=1),
    "quarterly": relativedelta(months=3),
    "yearly": relativedelta(years=1),
}
_MONTHS_PER_CYCLE = {"weekly": 12 / 52, "monthly": 1.0, "quarterly": 3.0, "yearly": 12.0}
CADENCES = tuple(_DELTAS)


@dataclass
class Subscription:
    """One recurring charge.

    ``amount`` is the charge per cycle (0 when unknown); ``cadence`` is one of
    :data:`CADENCES`; ``renews_on`` is a ``YYYY-MM-DD`` reference renewal date
    the next occurrence is computed from.
    """

    id: str
    name: str
    amount: float = 0.0
    currency: str = ""
    cadence: str = "monthly"
    renews_on: str = ""
    notes: str = ""
    created: str = ""
    updated: str = ""


def parse_date(value: str) -> date | None:
    """A ``date`` from an ISO date/datetime string (date part), or ``None``."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def next_renewal(sub: Subscription, today: date) -> date | None:
    """The next renewal on/after ``today`` computed from the anchor + cadence.

    ``None`` when there's no valid anchor date, or an unknown cadence whose
    anchor is already in the past (nothing to roll it forward by).
    """
    anchor = parse_date(sub.renews_on)
    if anchor is None:
        return None
    step = _DELTAS.get(sub.cadence)
    if step is None:
        return anchor if anchor >= today else None
    d = anchor
    guard = 0
    while d < today and guard < 5000:
        d = d + step
        guard += 1
    return d


def monthly_amount(sub: Subscription) -> float | None:
    """The charge normalized to a monthly figure, or ``None`` for an unknown cadence."""
    months = _MONTHS_PER_CYCLE.get(sub.cadence)
    if months is None:
        return None
    return sub.amount / months


def _coerce_amount(value: object, default: float = 0.0) -> float:
    try:
        return max(0.0, float(str(value).strip().replace(",", ".")))
    except (TypeError, ValueError):
        return default


def _normalize_cadence(value: str) -> str:
    value = (value or "").strip().lower()
    if value in ("annual", "annually", "year", "yearly"):
        return "yearly"
    if value in ("month", "monthly"):
        return "monthly"
    if value in ("week", "weekly"):
        return "weekly"
    if value in ("quarter", "quarterly"):
        return "quarterly"
    return value  # stored as-is; compute helpers return None for unknowns


def _normalize_date(value: str) -> str:
    d = parse_date(value)
    return d.isoformat() if d is not None else (value or "").strip()


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.subscriptions_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS subscriptions ("
        " id TEXT PRIMARY KEY, name TEXT NOT NULL, amount REAL DEFAULT 0,"
        " currency TEXT DEFAULT '', cadence TEXT DEFAULT 'monthly',"
        " renews_on TEXT DEFAULT '', notes TEXT DEFAULT '',"
        " created TEXT DEFAULT '', updated TEXT DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def _row_to_sub(row: sqlite3.Row) -> Subscription:
    return Subscription(
        id=row["id"],
        name=row["name"],
        amount=float(row["amount"] or 0.0),
        currency=row["currency"] or "",
        cadence=row["cadence"] or "monthly",
        renews_on=row["renews_on"] or "",
        notes=row["notes"] or "",
        created=row["created"] or "",
        updated=row["updated"] or "",
    )


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def create_subscription(
    settings: Settings,
    name: str,
    amount: object = 0,
    currency: str = "",
    cadence: str = "monthly",
    renews_on: str = "",
    notes: str = "",
) -> Subscription:
    """Insert a subscription and return it (with a generated id and timestamps)."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.create_subscription(
            settings, name, _coerce_amount(amount), currency, cadence, renews_on, notes
        )
    now = _stamp_now(settings)
    sub = Subscription(
        id=uuid.uuid4().hex[:12],
        name=name.strip(),
        amount=_coerce_amount(amount),
        currency=currency.strip(),
        cadence=_normalize_cadence(cadence) or "monthly",
        renews_on=_normalize_date(renews_on),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO subscriptions"
            " (id, name, amount, currency, cadence, renews_on, notes, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sub.id, sub.name, sub.amount, sub.currency, sub.cadence,
             sub.renews_on, sub.notes, sub.created, sub.updated),
        )
    return sub


def get_subscription(settings: Settings, sub_id: str) -> Subscription | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_subscription(settings, sub_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM subscriptions WHERE id = ?", (sub_id,)).fetchone()
    return _row_to_sub(row) if row else None


def list_subscriptions(settings: Settings) -> list[Subscription]:
    """All subscriptions, soonest next-renewal first (undated last, then by name)."""
    if storage_postgres := postgres_backend(settings):
        subs = storage_postgres.list_subscriptions(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM subscriptions").fetchall()
        subs = [_row_to_sub(r) for r in rows]
    today = date.today()

    def _key(sub: Subscription) -> tuple[int, str, str]:
        nxt = next_renewal(sub, today)
        return (0, nxt.isoformat(), sub.name.lower()) if nxt else (1, "", sub.name.lower())

    return sorted(subs, key=_key)


def update_subscription(settings: Settings, sub_id: str, **fields: object) -> Subscription | None:
    """Update a subscription's columns; return it, or ``None`` if absent."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.update_subscription(settings, sub_id, fields)
    existing = get_subscription(settings, sub_id)
    if existing is None:
        return None
    updates: dict[str, object] = {
        k: str(v).strip() for k, v in fields.items() if k in _TEXT_FIELDS and v is not None
    }
    if "cadence" in updates:
        updates["cadence"] = _normalize_cadence(str(updates["cadence"])) or existing.cadence
    if "renews_on" in updates:
        updates["renews_on"] = _normalize_date(str(updates["renews_on"]))
    if fields.get("amount") is not None:
        updates["amount"] = _coerce_amount(fields["amount"])
    if not updates:
        return existing
    updates["updated"] = _stamp_now(settings)
    columns = ", ".join(f"{k} = ?" for k in updates)
    with _connect(settings) as conn:
        conn.execute(
            f"UPDATE subscriptions SET {columns} WHERE id = ?", (*updates.values(), sub_id)
        )
    return get_subscription(settings, sub_id)


def delete_subscription(settings: Settings, sub_id: str) -> Subscription | None:
    """Delete a subscription by id; return it if it existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_subscription(settings, sub_id)
    existing = get_subscription(settings, sub_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
    return existing


def find_subscriptions(settings: Settings, query: str) -> list[Subscription]:
    """Candidates for ``query``: exact-id match alone, else name-substring matches."""
    query = query.strip()
    if not query:
        return []
    exact = get_subscription(settings, query)
    if exact is not None:
        return [exact]
    needle = query.lower()
    return [s for s in list_subscriptions(settings) if needle in s.name.lower()]


def find_subscription(settings: Settings, query: str) -> Subscription | None:
    matches = find_subscriptions(settings, query)
    return matches[0] if matches else None


def find_exact_name(settings: Settings, name: str) -> Subscription | None:
    """The subscription whose name exactly matches (case-insensitive), for dedupe."""
    needle = name.strip().lower()
    if not needle:
        return None
    for s in list_subscriptions(settings):
        if s.name.strip().lower() == needle:
            return s
    return None
