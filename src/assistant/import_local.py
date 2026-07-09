"""One-shot import from the local ``memory/`` SQLite + markdown stores into Postgres."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .calendar import store as calendar_store
from .config import Settings, get_settings
from .docs import store as docs_store
from .memory import store as memory_store
from .memory.embeddings import embed_passages
from .tasks import store as task_store
from . import storage_postgres

logger = logging.getLogger(__name__)


def _local_settings() -> Settings:
    return get_settings().model_copy(update={"storage_backend": "local"})


def _postgres_settings() -> Settings:
    settings = get_settings()
    if settings.storage_backend != "postgres" or not settings.database_url:
        raise RuntimeError("Set STORAGE_BACKEND=postgres and DATABASE_URL before importing")
    return settings


def _read_turn_count(local: Settings) -> int | None:
    conn = sqlite3.connect(local.memory_db_path)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'turn_count'").fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def _read_local_doc_chunks(local: Settings, doc_id: str) -> list[str]:
    conn = sqlite3.connect(local.docs_db_path)
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE doc_id = ? ORDER BY ord", (doc_id,)
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


def _read_telegram_chats(local: Settings) -> list[int]:
    path = local.memory_path / "telegram_chats.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(chat_id) for chat_id in data]


def import_notes(local: Settings, pg: Settings) -> int:
    count = 0
    for note in memory_store.list_notes(local):
        storage_postgres.write_note(pg, note)
        count += 1
    storage_postgres.reindex_memory(pg)
    return count


def import_calendar(local: Settings, pg: Settings) -> int:
    events = calendar_store.list_events(local)
    for event in events:
        storage_postgres.restore_event(pg, event)
    return len(events)


def import_tasks(local: Settings, pg: Settings) -> int:
    tasks = task_store.list_tasks(local)
    for task in tasks:
        storage_postgres.restore_task(pg, task)
    return len(tasks)


def import_documents(local: Settings, pg: Settings) -> int:
    count = 0
    for doc in docs_store.list_documents(local):
        pieces = _read_local_doc_chunks(local, doc.id)
        if not pieces:
            pieces = docs_store.chunk_text(doc.text, local.docs_chunk_chars)
        vectors = embed_passages(pieces, local) if pieces else []
        storage_postgres.ensure_docs_schema(pg)
        with storage_postgres.connect(pg) as conn:
            if vectors and storage_postgres.docs_meta_get(conn, "dim") is None:
                storage_postgres.docs_meta_set(conn, "dim", str(len(vectors[0])))
                storage_postgres.docs_meta_set(conn, "embedding_model", local.embedding_model)
            conn.execute(
                "INSERT INTO assistant_documents(id, title, text, added) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT(id) DO UPDATE SET "
                "title = excluded.title, text = excluded.text, added = excluded.added",
                (doc.id, doc.title, doc.text, doc.added),
            )
            conn.execute("DELETE FROM assistant_document_chunks WHERE doc_id = %s", (doc.id,))
            for ord_, (piece, vector) in enumerate(zip(pieces, vectors)):
                conn.execute(
                    "INSERT INTO assistant_document_chunks(doc_id, ord, text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (doc.id, ord_, piece, storage_postgres.vector_literal(vector)),
                )
        count += 1
    return count


def import_telegram(local: Settings, pg: Settings) -> int:
    chats = _read_telegram_chats(local)
    for chat_id in chats:
        storage_postgres.pair_telegram_chat(pg, chat_id)
    return len(chats)


def import_turn_counter(local: Settings, pg: Settings) -> None:
    turn_count = _read_turn_count(local)
    if turn_count is None:
        return
    storage_postgres.ensure_memory_schema(pg)
    with storage_postgres.connect(pg) as conn:
        storage_postgres._meta_set(conn, "turn_count", str(turn_count))


def import_checkpoints(local: Settings, pg: Settings) -> int:
    src = SqliteSaver(sqlite3.connect(local.checkpoints_db_path, check_same_thread=False))
    dst_conn = psycopg.connect(storage_postgres.require_url(pg), autocommit=True)
    dst = PostgresSaver(dst_conn)
    dst.setup()

    raw = sqlite3.connect(local.checkpoints_db_path)
    try:
        threads = [row[0] for row in raw.execute("SELECT DISTINCT thread_id FROM checkpoints")]
    finally:
        raw.close()

    copied = 0
    for thread_id in threads:
        config = {"configurable": {"thread_id": thread_id}}
        for tup in src.list(config):
            dst.put(
                tup.config,
                tup.checkpoint,
                tup.metadata,
                tup.checkpoint.get("channel_versions") or {},
            )
            copied += 1
    dst_conn.close()
    return copied


def import_all(local: Settings | None = None, pg: Settings | None = None) -> dict[str, int]:
    local = local or _local_settings()
    pg = pg or _postgres_settings()
    local.memory_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "notes": import_notes(local, pg),
        "calendar_events": import_calendar(local, pg),
        "tasks": import_tasks(local, pg),
        "documents": import_documents(local, pg),
        "telegram_chats": import_telegram(local, pg),
        "checkpoints": import_checkpoints(local, pg),
    }
    import_turn_counter(local, pg)
    storage_postgres.regenerate_index(pg)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = import_all()
    except Exception:
        logger.exception("import failed")
        sys.exit(1)
    for key, value in summary.items():
        logger.info("%s: %s", key, value)
    logger.info("import complete")


if __name__ == "__main__":
    main()
