"""Follow-up store for the Postgres backend.

Mirrors :mod:`assistant.followups` (the assistant's own initiative store) so that
under ``STORAGE_BACKEND=postgres`` follow-ups live in Neon, not a prod-only
``followups.db``. Timestamps are stored as ISO-8601 TEXT, byte-identical to the
sqlite store, so ``parse_dt`` reads them the same and a claim is exactly-once via
``WHERE status = 'open'`` — the same claim-first discipline.
"""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect


def ensure_followups_schema(settings: Settings) -> None:
    if _schema_done(settings, "followups"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_followups (
              id TEXT PRIMARY KEY,
              due TEXT NOT NULL,
              topic TEXT NOT NULL,
              context TEXT NOT NULL DEFAULT '',
              thread_id TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'open',
              fired_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "followups")


def _from_row(row: dict):
    from ..followups import Followup

    return Followup(
        id=str(row["id"]),
        due=str(row["due"]),
        topic=str(row["topic"]),
        context=str(row.get("context") or ""),
        thread_id=str(row.get("thread_id") or ""),
        created_at=str(row.get("created_at") or ""),
        status=str(row.get("status") or "open"),
        fired_at=str(row.get("fired_at") or ""),
    )


def add_followup(settings: Settings, followup) -> None:
    ensure_followups_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_followups"
            " (id, due, topic, context, thread_id, created_at, status, fired_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, 'open', '')",
            (
                followup.id,
                followup.due,
                followup.topic,
                followup.context,
                followup.thread_id,
                followup.created_at,
            ),
        )


def list_open_followups(settings: Settings) -> list:
    ensure_followups_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(
                "SELECT * FROM assistant_followups WHERE status = 'open' ORDER BY due"
            )
        )
    return [_from_row(row) for row in rows]


def cancel_followup(settings: Settings, ident: str) -> bool:
    ensure_followups_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "UPDATE assistant_followups SET status = 'cancelled'"
            " WHERE id = %s AND status = 'open' RETURNING id",
            (ident,),
        )
        return cur.fetchone() is not None


def update_followup(
    settings: Settings, ident: str, due: str, topic: str, context: str
) -> bool:
    ensure_followups_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "UPDATE assistant_followups SET due = %s, topic = %s, context = %s"
            " WHERE id = %s AND status = 'open' RETURNING id",
            (due, topic, context, ident),
        )
        return cur.fetchone() is not None


def claim_due_followups(settings: Settings, fired_at: str, current) -> list:
    from ..calendar.store import parse_dt

    ensure_followups_schema(settings)
    claimed: list = []
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(
                "SELECT * FROM assistant_followups WHERE status = 'open' ORDER BY due"
            )
        )
        for row in rows:
            due = parse_dt(str(row["due"]))
            if due is None or due > current:
                continue
            cur = conn.execute(
                "UPDATE assistant_followups SET status = 'fired', fired_at = %s"
                " WHERE id = %s AND status = 'open' RETURNING id",
                (fired_at, row["id"]),
            )
            if cur.fetchone() is not None:
                claimed.append(_from_row(row))
    return claimed
