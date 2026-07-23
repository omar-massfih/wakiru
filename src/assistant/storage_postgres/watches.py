"""Watch store for the Postgres backend.

Mirrors :mod:`assistant.watches` (model-registered perception) so that under
``STORAGE_BACKEND=postgres`` watches live in Neon, not a prod-only
``followups.db``. Timestamps are ISO-8601 TEXT, byte-identical to the sqlite
store; claiming a one-shot firing is exactly-once via ``WHERE status =
'active'``, the followups discipline.
"""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect


def ensure_watches_schema(settings: Settings) -> None:
    if _schema_done(settings, "watches"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_watches (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              pattern TEXT NOT NULL,
              note TEXT NOT NULL DEFAULT '',
              lead_minutes INTEGER NOT NULL DEFAULT 30,
              repeat INTEGER NOT NULL DEFAULT 0,
              fire_at TEXT NOT NULL DEFAULT '',
              expires_at TEXT NOT NULL DEFAULT '',
              last_match_hash TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'active',
              url TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # Tables created before a column existed need it added in place.
        conn.execute(
            "ALTER TABLE assistant_watches"
            " ADD COLUMN IF NOT EXISTS url TEXT NOT NULL DEFAULT ''"
        )
    _schema_mark(settings, "watches")


def _from_row(row: dict):
    from ..watches import Watch

    return Watch(
        id=str(row["id"]),
        kind=str(row["kind"]),
        pattern=str(row["pattern"]),
        note=str(row.get("note") or ""),
        lead_minutes=int(row.get("lead_minutes") or 0),
        repeat=bool(row.get("repeat")),
        fire_at=str(row.get("fire_at") or ""),
        url=str(row.get("url") or ""),
        expires_at=str(row.get("expires_at") or ""),
        last_match_hash=str(row.get("last_match_hash") or ""),
        created_at=str(row.get("created_at") or ""),
        status=str(row.get("status") or "active"),
    )


def add_watch(settings: Settings, watch) -> None:
    ensure_watches_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_watches"
            " (id, kind, pattern, note, lead_minutes, repeat, fire_at, url,"
            "  expires_at, last_match_hash, created_at, status)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')",
            (
                watch.id,
                watch.kind,
                watch.pattern,
                watch.note,
                watch.lead_minutes,
                int(watch.repeat),
                watch.fire_at,
                watch.url,
                watch.expires_at,
                watch.last_match_hash,
                watch.created_at,
            ),
        )


def list_active_watches(settings: Settings, stamp: str, current) -> list:
    from ..calendar.store import parse_dt

    ensure_watches_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute("SELECT * FROM assistant_watches WHERE status = 'active'")
        )
        expired = [
            str(row["id"])
            for row in rows
            if (until := parse_dt(str(row["expires_at"]))) is not None
            and until <= current
        ]
        for wid in expired:
            conn.execute(
                "UPDATE assistant_watches SET status = 'expired'"
                " WHERE id = %s AND status = 'active'",
                (wid,),
            )
    dropped = set(expired)
    return [_from_row(row) for row in rows if str(row["id"]) not in dropped]


def cancel_watch(settings: Settings, ident: str) -> bool:
    ensure_watches_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "UPDATE assistant_watches SET status = 'cancelled'"
            " WHERE id = %s AND status = 'active' RETURNING id",
            (ident,),
        )
        return cur.fetchone() is not None


def claim_watch(settings: Settings, ident: str, repeat: bool, match_hash: str) -> bool:
    ensure_watches_schema(settings)
    with connect(settings) as conn:
        if repeat:
            cur = conn.execute(
                "UPDATE assistant_watches SET last_match_hash = %s"
                " WHERE id = %s AND status = 'active' AND last_match_hash != %s"
                " RETURNING id",
                (match_hash, ident, match_hash),
            )
        else:
            cur = conn.execute(
                "UPDATE assistant_watches SET status = 'fired', last_match_hash = %s"
                " WHERE id = %s AND status = 'active' RETURNING id",
                (match_hash, ident),
            )
        return cur.fetchone() is not None
