"""Small namespaced key/value store for the Postgres backend.

Backs the odds and ends that were prod-only single-file state under the local
backend: the heartbeat's wake/push/mail-hash state and its self-paced-wake keys,
the nightly sleep pass's last-LLM-pass stamp, and the cached unread-mail snapshot
(previously ``mail_snapshot.json``). One table, keyed by ``(namespace, key)``, so
each caller carves out its own namespace.
"""

from __future__ import annotations

from ..config import Settings
from .core import _executemany, _schema_done, _schema_mark, connect


def ensure_kv_schema(settings: Settings) -> None:
    if _schema_done(settings, "kv"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_kv (
              namespace TEXT NOT NULL,
              key TEXT NOT NULL,
              value TEXT NOT NULL,
              PRIMARY KEY (namespace, key)
            )
            """
        )
    _schema_mark(settings, "kv")


def kv_get(settings: Settings, namespace: str, key: str) -> str:
    ensure_kv_schema(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT value FROM assistant_kv WHERE namespace = %s AND key = %s",
            (namespace, key),
        ).fetchone()
    return str(row[0]) if row else ""


def kv_set(settings: Settings, namespace: str, key: str, value: str) -> None:
    ensure_kv_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_kv (namespace, key, value) VALUES (%s, %s, %s)"
            " ON CONFLICT (namespace, key) DO UPDATE SET value = EXCLUDED.value",
            (namespace, key, value),
        )


def kv_clear(settings: Settings, namespace: str, keys: list[str]) -> None:
    if not keys:
        return
    ensure_kv_schema(settings)
    with connect(settings) as conn:
        _executemany(
            conn,
            "DELETE FROM assistant_kv WHERE namespace = %s AND key = %s",
            [(namespace, key) for key in keys],
        )
