"""Subscriptions table for the Postgres backend (twin of assistant.subscriptions.store)."""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect

_COLS = "id, name, amount, currency, cadence, renews_on, notes, created, updated"


def ensure_subscriptions_schema(settings: Settings) -> None:
    if _schema_done(settings, "subscriptions"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_subscriptions (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              amount DOUBLE PRECISION NOT NULL DEFAULT 0,
              currency TEXT NOT NULL DEFAULT '',
              cadence TEXT NOT NULL DEFAULT 'monthly',
              renews_on TEXT NOT NULL DEFAULT '',
              notes TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "subscriptions")


def _sub_from_row(row: dict):
    from ..subscriptions.store import Subscription

    return Subscription(
        id=str(row["id"]),
        name=str(row["name"]),
        amount=float(row.get("amount") or 0.0),
        currency=str(row.get("currency") or ""),
        cadence=str(row.get("cadence") or "monthly"),
        renews_on=str(row.get("renews_on") or ""),
        notes=str(row.get("notes") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
    )


def create_subscription(
    settings: Settings,
    name: str,
    amount: float = 0.0,
    currency: str = "",
    cadence: str = "monthly",
    renews_on: str = "",
    notes: str = "",
):
    import uuid

    from ..subscriptions import store as sub_store

    ensure_subscriptions_schema(settings)
    now = sub_store._stamp_now(settings)
    sub = sub_store.Subscription(
        id=uuid.uuid4().hex[:12],
        name=name.strip(),
        amount=sub_store._coerce_amount(amount),
        currency=currency.strip(),
        cadence=sub_store._normalize_cadence(cadence) or "monthly",
        renews_on=sub_store._normalize_date(renews_on),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_subscriptions"
            " (id, name, amount, currency, cadence, renews_on, notes, created, updated)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (sub.id, sub.name, sub.amount, sub.currency, sub.cadence,
             sub.renews_on, sub.notes, sub.created, sub.updated),
        )
    return sub


def get_subscription(settings: Settings, sub_id: str):
    ensure_subscriptions_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(f"SELECT {_COLS} FROM assistant_subscriptions WHERE id = %s", (sub_id,))
        )
    return _sub_from_row(rows[0]) if rows else None


def list_subscriptions(settings: Settings):
    ensure_subscriptions_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_subscriptions"))
    return [_sub_from_row(r) for r in rows]


def update_subscription(settings: Settings, sub_id: str, fields: dict):
    from ..subscriptions import store as sub_store

    ensure_subscriptions_schema(settings)
    existing = get_subscription(settings, sub_id)
    if existing is None:
        return None
    updates: dict[str, object] = {
        k: str(v).strip()
        for k, v in fields.items()
        if k in sub_store._TEXT_FIELDS and v is not None
    }
    if "cadence" in updates:
        updates["cadence"] = sub_store._normalize_cadence(str(updates["cadence"])) or existing.cadence
    if "renews_on" in updates:
        updates["renews_on"] = sub_store._normalize_date(str(updates["renews_on"]))
    if fields.get("amount") is not None:
        updates["amount"] = sub_store._coerce_amount(fields["amount"])
    if not updates:
        return existing
    updates["updated"] = sub_store._stamp_now(settings)
    assignments = ", ".join(f"{k} = %s" for k in updates)
    with connect(settings) as conn:
        conn.execute(
            f"UPDATE assistant_subscriptions SET {assignments} WHERE id = %s",
            (*updates.values(), sub_id),
        )
    return get_subscription(settings, sub_id)


def delete_subscription(settings: Settings, sub_id: str):
    existing = get_subscription(settings, sub_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_subscriptions WHERE id = %s", (sub_id,))
    return existing
