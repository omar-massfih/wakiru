"""Connection, pool, and schema plumbing for the Postgres backend.

One submodule per domain mirrors the sqlite stores (memory, docs, calendar,
tasks, ledgers, mutes, telegram); this module is what they all share. The
package as a whole keeps the surface the old single-module storage_postgres
exposed — callers still ``from .. import storage_postgres``.
"""

from __future__ import annotations

import atexit
import threading
from collections.abc import Sequence
from contextlib import contextmanager

from ..config import Settings


def enabled(settings: Settings) -> bool:
    return settings.storage_backend == "postgres"


def require_url(settings: Settings) -> str:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
    return settings.database_url


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised in deployed config
        raise RuntimeError(
            "STORAGE_BACKEND=postgres requires the 'psycopg[binary]' dependency"
        ) from exc
    return psycopg


# One pool per DSN for the process (opening a fresh connection per operation
# costs a TLS round-trip each time on serverless Postgres). check= revalidates
# a connection on checkout — an idle one dies when e.g. Neon suspends — the
# same discipline as the checkpointer pool in agent._checkpointer.
_pools: dict[str, object] = {}
_pools_lock = threading.Lock()
# (dsn, schema) pairs whose CREATE TABLE IF NOT EXISTS pass already ran in
# this process; the DDL is idempotent but not free, and it used to run on
# nearly every operation.
_ensured_schemas: set[tuple[str, str]] = set()


def _pool(settings: Settings):
    _psycopg()  # surface the missing-dependency error first
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:  # pragma: no cover - ships with the postgres extra
        raise RuntimeError(
            "STORAGE_BACKEND=postgres requires the 'psycopg-pool' dependency"
        ) from exc

    dsn = require_url(settings)
    with _pools_lock:
        pool = _pools.get(dsn)
        if pool is None:
            pool = ConnectionPool(
                dsn,
                min_size=0,
                max_size=4,
                open=True,
                check=ConnectionPool.check_connection,
            )
            atexit.register(pool.close)
            _pools[dsn] = pool
    return pool


@contextmanager
def connect(settings: Settings):
    # pool.connection() commits on clean exit, rolls back on exception, and
    # returns the connection to the pool — the same transaction contract the
    # previous one-connection-per-call implementation provided.
    with _pool(settings).connection() as conn:
        yield conn


def _schema_done(settings: Settings, schema: str) -> bool:
    return (require_url(settings), schema) in _ensured_schemas


def _schema_mark(settings: Settings, schema: str) -> None:
    # Marked only after the DDL succeeded; concurrent first calls may both run
    # the CREATE IF NOT EXISTS pass, which is harmless.
    _ensured_schemas.add((require_url(settings), schema))


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.9g}" for v in vector) + "]"


def _rows(cur) -> list[dict]:
    cols = [col.name for col in cur.description or []]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _executemany(conn, sql: str, params: list) -> None:
    if not params:
        return
    with conn.cursor() as cur:
        cur.executemany(sql, params)

