"""Postgres backend routing tests.

These keep CI database-free while still proving that STORAGE_BACKEND=postgres
uses the Postgres adapters instead of the local markdown/sqlite stores.
"""

from __future__ import annotations

import pytest

from assistant.agent import _checkpointer
from assistant.config import Settings
from assistant.docs import store as docs_store
from assistant.memory import index, store
from assistant.memory.store import Note


def test_storage_backend_defaults_to_local() -> None:
    settings = Settings()
    assert settings.storage_backend == "local"
    assert settings.database_url is None


def test_postgres_note_store_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")
    note = Note(name="likes-tea", description="Likes tea", body="Omar likes tea.")
    calls: list[tuple[str, object]] = []

    from assistant import storage_postgres

    monkeypatch.setattr(storage_postgres, "write_note", lambda _s, n: calls.append(("write", n)) or store.note_path(_s, n))
    monkeypatch.setattr(storage_postgres, "find_note", lambda _s, name: calls.append(("find", name)) or note)
    monkeypatch.setattr(storage_postgres, "list_notes", lambda _s: calls.append(("list", None)) or [note])

    assert store.write_note(settings, note).name == "likes-tea.md"
    assert store.find_note(settings, "likes-tea") == note
    assert store.list_notes(settings) == [note]
    assert calls == [("write", note), ("find", "likes-tea"), ("list", None)]


def test_postgres_memory_index_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres

    monkeypatch.setattr(
        storage_postgres,
        "search_memory_index",
        lambda _settings, vector, k: [("n", "p", "d", "semantic", 0.5, 1, "", 0.9)],
    )
    monkeypatch.setattr(storage_postgres, "bump_turn_counter", lambda _settings: 42)

    assert index.search_ranked(settings, [0.1, 0.2], 3)[0][0] == "n"
    assert index.bump_turn_counter(settings) == 42


def test_postgres_docs_store_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")
    doc = docs_store.Document(id="doc1", title="Title", text="Body", chunks=1)

    from assistant import storage_postgres

    monkeypatch.setattr(docs_store, "embed_passages", lambda pieces, _settings: [[0.1, 0.2] for _ in pieces])
    monkeypatch.setattr(
        storage_postgres,
        "add_document",
        lambda _settings, title, text, pieces, vectors: doc,
    )
    monkeypatch.setattr(storage_postgres, "list_documents", lambda _settings: [doc])

    assert docs_store.add_document(settings, "Title", "Body") == doc
    assert docs_store.list_documents(settings) == [doc]


def test_postgres_checkpointer_requires_database_url() -> None:
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        _checkpointer(Settings(storage_backend="postgres", database_url=None))


def test_postgres_checkpointer_uses_a_health_checked_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpointer must ride a pool that revalidates connections on checkout —
    a single long-lived connection dies when serverless Postgres suspends."""
    import langgraph.checkpoint.postgres as lg_postgres
    import psycopg_pool

    captured: dict = {}

    class FakePool:
        def __init__(self, conninfo: str, **kwargs):
            captured["conninfo"] = conninfo
            captured.update(kwargs)

        @staticmethod
        def check_connection(conn) -> None:  # referenced as check=
            pass

        def close(self) -> None:  # registered via atexit
            pass

    class FakeSaver:
        def __init__(self, conn):
            captured["conn"] = conn

        def setup(self) -> None:
            captured["setup"] = True

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", FakePool)
    monkeypatch.setattr(lg_postgres, "PostgresSaver", FakeSaver)

    saver = _checkpointer(Settings(storage_backend="postgres", database_url="postgres://example"))

    assert isinstance(saver, FakeSaver)
    assert isinstance(captured["conn"], FakePool)
    assert captured["setup"] is True
    assert captured["check"] is FakePool.check_connection
    assert captured["kwargs"]["autocommit"] is True
    assert captured["kwargs"]["row_factory"] is not None



def test_postgres_calendar_and_task_stores_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres
    from assistant.calendar import store as calendar_store
    from assistant.tasks import store as task_store

    event = calendar_store.Event(id="e1", title="Dentist", start="2026-07-09T12:00:00+02:00")
    task = task_store.Task(id="t1", title="Pay bill")

    monkeypatch.setattr(storage_postgres, "create_event", lambda *args: event)
    monkeypatch.setattr(storage_postgres, "list_events", lambda _settings: [event])
    monkeypatch.setattr(storage_postgres, "update_event", lambda _settings, event_id, fields: event)
    monkeypatch.setattr(storage_postgres, "create_task", lambda *args: task)
    monkeypatch.setattr(storage_postgres, "list_tasks", lambda _settings: [task])
    monkeypatch.setattr(storage_postgres, "complete_task", lambda _settings, task_id: task)

    assert calendar_store.create_event(settings, "Dentist", event.start) == event
    assert calendar_store.list_events(settings) == [event]
    assert calendar_store.update_event(settings, "e1", title="New") == event
    assert task_store.create_task(settings, "Pay bill") == task
    assert task_store.list_tasks(settings) == [task]
    assert task_store.complete_task(settings, "t1") == task


def test_postgres_undo_ledgers_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres
    from assistant.calendar import undo as calendar_undo
    from assistant.tasks import undo as task_undo

    calls: list[str] = []
    monkeypatch.setattr(storage_postgres, "record_calendar_write", lambda *args: calls.append("calendar"))
    monkeypatch.setattr(storage_postgres, "record_task_write", lambda *args: calls.append("task"))

    calendar_undo.record_write(settings, "thread", "batch", "event", "create", "summary", None)
    task_undo.record_write(settings, "thread", "batch", "task", "add", "summary", None)

    assert calls == ["calendar", "task"]


def test_postgres_reminder_ledgers_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres
    from assistant.calendar import reminders as calendar_reminders
    from assistant.tasks import reminders as task_reminders

    monkeypatch.setattr(calendar_reminders, "due_reminders", lambda _settings, current=None: [{"event_id": "e1", "start": "s", "covered_leads": [60], "message": "event"}])
    monkeypatch.setattr(task_reminders, "due_task_reminders", lambda _settings, current=None: [{"task_id": "t1", "due": "d", "covered_leads": [60], "message": "task"}])
    monkeypatch.setattr(storage_postgres, "claim_calendar_reminders", lambda _settings, due, fired_at, current: due)
    monkeypatch.setattr(storage_postgres, "claim_task_reminders", lambda _settings, due, fired_at, current: due)
    monkeypatch.setattr(calendar_reminders, "deliver_reminder", lambda _settings, reminder: None)
    monkeypatch.setattr(task_reminders, "deliver_reminder", lambda _settings, reminder: None)

    assert calendar_reminders.run_reminders(settings)[0]["message"] == "event"
    assert task_reminders.run_task_reminders(settings)[0]["message"] == "task"



def test_postgres_telegram_pairing_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres, telegram

    paired: list[int] = []
    monkeypatch.setattr(storage_postgres, "paired_telegram_chats", lambda _settings: list(paired))
    monkeypatch.setattr(storage_postgres, "pair_telegram_chat", lambda _settings, chat_id: paired.append(chat_id))

    assert telegram._paired_chats(settings) == []
    telegram._pair(settings, 123)
    assert telegram._paired_chats(settings) == [123]
