"""Mailbox-mutation audit ledger for the Postgres backend.

Mirrors :mod:`assistant.mail.audit`'s sqlite table so that under
``STORAGE_BACKEND=postgres`` the heartbeat's triage accountability survives
redeploys, not just restarts.
"""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect


def ensure_mail_schema(settings: Settings) -> None:
    if _schema_done(settings, "mail"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_mail_audit (
              id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
              actor TEXT NOT NULL,
              action TEXT NOT NULL,
              uid TEXT NOT NULL,
              detail TEXT NOT NULL,
              at TEXT NOT NULL
            )
            """
        )
    _schema_mark(settings, "mail")


def record_mail_audit(
    settings: Settings, actor: str, action: str, uid: str, detail: str, at: str
) -> None:
    ensure_mail_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_mail_audit (actor, action, uid, detail, at)"
            " VALUES (%s, %s, %s, %s, %s)",
            (actor, action, uid, detail, at),
        )


def mail_audit_rows(
    settings: Settings, limit: int = 5, actor: str | None = None
) -> list[dict]:
    ensure_mail_schema(settings)
    with connect(settings) as conn:
        if actor is None:
            cur = conn.execute(
                "SELECT * FROM assistant_mail_audit ORDER BY id DESC LIMIT %s",
                (limit,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM assistant_mail_audit WHERE actor = %s"
                " ORDER BY id DESC LIMIT %s",
                (actor, limit),
            )
        return _rows(cur)
