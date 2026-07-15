"""Postgres/pgvector storage helpers for Vercel Marketplace databases.

The local backend remains the default. When ``STORAGE_BACKEND=postgres`` this
package provides the durable stores that replace the markdown/sqlite files
under ``memory/`` with Neon Postgres tables, while preserving the public
store/index APIs used by the rest of the assistant.

One submodule per domain, mirroring the sqlite stores; :mod:`.core` holds the
shared pool/schema plumbing. Everything callers use is re-exported here, so
the split away from the old single module changed no import site — and tests
monkeypatch these package attributes, which every caller (including the
ledger drivers' ``getattr``) resolves at call time.
"""

from __future__ import annotations

from .calendar import (
    create_event,
    delete_event,
    ensure_calendar_schema,
    get_event,
    list_events,
    restore_event,
    update_event,
)
from .core import connect, enabled, require_url, vector_literal
from .docs import (
    add_document,
    delete_document,
    docs_meta_get,
    docs_meta_set,
    ensure_docs_schema,
    get_document,
    list_documents,
    reindex_docs,
    reindex_documents,
    search_chunks,
)
from .followups import (
    add_followup,
    cancel_followup,
    claim_due_followups,
    ensure_followups_schema,
    list_open_followups,
    update_followup,
)
from .kv import ensure_kv_schema, kv_clear, kv_get, kv_set
from .ledgers import (
    calendar_write_rows,
    claim_calendar_reminders,
    claim_fired,
    claim_task_reminders,
    ensure_fired_schema,
    mark_calendar_writes_undone,
    mark_task_writes_undone,
    record_calendar_write,
    record_task_write,
    task_write_rows,
)
from .memory import (
    bump_recall,
    bump_turn_counter,
    delete_note,
    ensure_memory_schema,
    find_note,
    get_stats,
    list_memory_entries,
    list_notes,
    meta_get,
    meta_set,
    prune_trash,
    purge_stale_files,
    read_index,
    regenerate_index,
    reindex_memory,
    remove_memory_index,
    search_memory_index,
    set_salience,
    unique_name,
    upsert_memory_index,
    virtual_note_path,
    write_note,
)
from .mutes import clear_mute, ensure_mutes_schema, list_mutes, set_mute
from .tasks import (
    complete_task,
    create_task,
    delete_task,
    ensure_tasks_schema,
    get_task,
    list_tasks,
    restore_task,
    update_task,
)
from .telegram import ensure_telegram_schema, pair_telegram_chat, paired_telegram_chats
from .threads import ensure_threads_schema, known_threads, touch_thread

__all__ = [
    "add_document",
    "add_followup",
    "bump_recall",
    "bump_turn_counter",
    "calendar_write_rows",
    "cancel_followup",
    "claim_calendar_reminders",
    "claim_due_followups",
    "claim_fired",
    "claim_task_reminders",
    "clear_mute",
    "complete_task",
    "connect",
    "create_event",
    "create_task",
    "delete_document",
    "delete_event",
    "delete_note",
    "delete_task",
    "docs_meta_get",
    "docs_meta_set",
    "enabled",
    "ensure_calendar_schema",
    "ensure_docs_schema",
    "ensure_fired_schema",
    "ensure_followups_schema",
    "ensure_kv_schema",
    "ensure_memory_schema",
    "ensure_mutes_schema",
    "ensure_tasks_schema",
    "ensure_telegram_schema",
    "ensure_threads_schema",
    "find_note",
    "get_document",
    "get_event",
    "get_stats",
    "get_task",
    "known_threads",
    "kv_clear",
    "kv_get",
    "kv_set",
    "list_documents",
    "list_events",
    "list_memory_entries",
    "list_mutes",
    "list_notes",
    "list_open_followups",
    "list_tasks",
    "mark_calendar_writes_undone",
    "mark_task_writes_undone",
    "meta_get",
    "meta_set",
    "pair_telegram_chat",
    "paired_telegram_chats",
    "prune_trash",
    "purge_stale_files",
    "read_index",
    "record_calendar_write",
    "record_task_write",
    "regenerate_index",
    "reindex_docs",
    "reindex_documents",
    "reindex_memory",
    "remove_memory_index",
    "require_url",
    "restore_event",
    "restore_task",
    "search_chunks",
    "search_memory_index",
    "set_mute",
    "set_salience",
    "task_write_rows",
    "touch_thread",
    "unique_name",
    "update_event",
    "update_followup",
    "update_task",
    "upsert_memory_index",
    "vector_literal",
    "virtual_note_path",
    "write_note",
]
