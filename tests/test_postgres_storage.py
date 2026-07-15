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



def test_postgres_connect_pools_and_schema_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """connect() must reuse one pool per DSN, and the CREATE TABLE pass must run
    once per process, not on every operation."""
    from contextlib import contextmanager

    import psycopg_pool

    from assistant import storage_postgres

    executed: list[str] = []

    class FakeConn:
        def execute(self, sql, *args):
            executed.append(sql)

    pools_created: list[str] = []

    class FakePool:
        def __init__(self, conninfo: str, **kwargs):
            pools_created.append(conninfo)

        @staticmethod
        def check_connection(conn) -> None:
            pass

        def close(self) -> None:
            pass

        @contextmanager
        def connection(self):
            yield FakeConn()

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", FakePool)
    # Process-wide pool/schema state lives on the core submodule.
    monkeypatch.setattr(storage_postgres.core, "_pools", {})
    monkeypatch.setattr(storage_postgres.core, "_ensured_schemas", set())

    settings = Settings(storage_backend="postgres", database_url="postgres://example")
    with storage_postgres.connect(settings):
        pass
    with storage_postgres.connect(settings):
        pass
    assert pools_created == ["postgres://example"]

    storage_postgres.ensure_tasks_schema(settings)
    first_pass = len(executed)
    assert first_pass > 0
    storage_postgres.ensure_tasks_schema(settings)
    assert len(executed) == first_pass


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


def test_postgres_task_undo_reverts_latest_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the Postgres branch of tasks undo_latest must revert the batch
    and return the user-facing summary, not silently no-op (mis-pasted guard)."""
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres
    from assistant.calendar.context import now
    from assistant.tasks import store as task_store
    from assistant.tasks import undo as task_undo

    applied_at = now(settings).isoformat(timespec="seconds")
    rows = [
        {
            "id": 7, "thread_id": "thread", "batch_id": "b1", "task_id": "t1",
            "op": "add", "summary": "added: Pay bill", "before_json": None,
            "applied_at": applied_at, "undone_at": None,
        }
    ]
    undone: list[list[int]] = []
    monkeypatch.setattr(storage_postgres, "task_write_rows", lambda _s, thread_id: rows)
    monkeypatch.setattr(
        storage_postgres, "delete_task",
        lambda _s, task_id: task_store.Task(id=task_id, title="Pay bill"),
    )
    monkeypatch.setattr(
        storage_postgres, "mark_task_writes_undone",
        lambda _s, ids, at: undone.append(ids),
    )

    assert task_undo.undo_latest(settings, "thread", 15) == "Undone: removed: Pay bill"
    assert undone == [[7]]

    monkeypatch.setattr(storage_postgres, "task_write_rows", lambda _s, thread_id: [])
    assert task_undo.undo_latest(settings, "thread", 15) == "Nothing to undo."


def test_postgres_calendar_undo_reverts_latest_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror of the tasks test so the calendar twin stays pinned too."""
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres
    from assistant.calendar import store as calendar_store
    from assistant.calendar import undo as calendar_undo
    from assistant.calendar.context import now

    applied_at = now(settings).isoformat(timespec="seconds")
    rows = [
        {
            "id": 3, "thread_id": "thread", "batch_id": "b1", "event_id": "e1",
            "op": "create", "summary": "created: Dentist", "before_json": None,
            "applied_at": applied_at, "undone_at": None,
        }
    ]
    undone: list[list[int]] = []
    monkeypatch.setattr(storage_postgres, "calendar_write_rows", lambda _s, thread_id: rows)
    monkeypatch.setattr(
        storage_postgres, "delete_event",
        lambda _s, event_id: calendar_store.Event(
            id=event_id, title="Dentist", start="2026-07-09T12:00:00+02:00"
        ),
    )
    monkeypatch.setattr(
        storage_postgres, "mark_calendar_writes_undone",
        lambda _s, ids, at: undone.append(ids),
    )

    assert calendar_undo.undo_latest(settings, "thread", 15) == "Undone: removed: Dentist"
    assert undone == [[3]]


def test_postgres_reminder_ledgers_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres
    from assistant.calendar import reminders as calendar_reminders
    from assistant.tasks import reminders as task_reminders

    monkeypatch.setattr(calendar_reminders, "due_reminders", lambda _settings, current=None: [{"event_id": "e1", "title": "Event", "start": "s", "covered_leads": [60], "message": "event"}])
    monkeypatch.setattr(task_reminders, "due_task_reminders", lambda _settings, current=None: [{"task_id": "t1", "title": "Task", "due": "d", "covered_leads": [60], "message": "task"}])
    monkeypatch.setattr(storage_postgres, "claim_calendar_reminders", lambda _settings, due, fired_at, current: due)
    monkeypatch.setattr(storage_postgres, "claim_task_reminders", lambda _settings, due, fired_at, current: due)
    monkeypatch.setattr(storage_postgres, "list_mutes", lambda _settings: [])
    monkeypatch.setattr("assistant.compose.compose_push", lambda s, **kw: kw["fallback"])
    monkeypatch.setattr(calendar_reminders, "deliver_reminder", lambda _settings, reminder: None)
    monkeypatch.setattr(task_reminders, "deliver_reminder", lambda _settings, reminder: None)

    assert calendar_reminders.run_reminders(settings)[0]["message"] == "event"
    assert task_reminders.run_task_reminders(settings)[0]["message"] == "task"



def test_postgres_mutes_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from datetime import timedelta

    from assistant import mutes, storage_postgres
    from assistant.calendar.context import now

    current = now(settings)
    until = current + timedelta(hours=1)

    calls: list[tuple] = []
    monkeypatch.setattr(
        storage_postgres, "set_mute",
        lambda _s, scope, target_id, u, reason, c: calls.append(("set", scope, target_id)),
    )
    monkeypatch.setattr(
        storage_postgres, "clear_mute",
        lambda _s, scope, target_id: calls.append(("clear", scope, target_id)) or True,
    )
    monkeypatch.setattr(
        storage_postgres, "list_mutes",
        lambda _s: [("event", "e1", until.isoformat(timespec="seconds"))],
    )

    mutes.set_mute(settings, "event", "e1", until, current=current)
    assert mutes.clear_mute(settings, "event", "e1") is True
    assert calls == [("set", "event", "e1"), ("clear", "event", "e1")]

    active = mutes.active_mutes(settings, current)
    assert set(active) == {("event", "e1")}
    # Expired rows returned by the backend are filtered out client-side.
    assert mutes.active_mutes(settings, until + timedelta(minutes=1)) == {}
    # And the due-list filter consumes the same view.
    due = [{"event_id": "e1", "message": "m"}, {"event_id": "e2", "message": "m2"}]
    assert mutes.filter_muted(settings, due, current, "event") == [due[1]]


def test_postgres_telegram_pairing_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(storage_backend="postgres", database_url="postgres://example")

    from assistant import storage_postgres, telegram

    paired: list[int] = []
    monkeypatch.setattr(storage_postgres, "paired_telegram_chats", lambda _settings: list(paired))
    monkeypatch.setattr(storage_postgres, "pair_telegram_chat", lambda _settings, chat_id: paired.append(chat_id))

    assert telegram._paired_chats(settings) == []
    telegram._pair(settings, 123)
    assert telegram._paired_chats(settings) == [123]


def _pg_settings() -> Settings:
    return Settings(storage_backend="postgres", database_url="postgres://example")


def test_postgres_followups_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    from assistant import followups, storage_postgres
    from assistant.calendar.context import now

    settings = _pg_settings()
    current = now(settings)
    row = followups.Followup(
        id="f1", due=(current + timedelta(hours=1)).isoformat(timespec="seconds"),
        topic="ask about the viewing", context="on Storgata",
    )
    calls: list[str] = []
    monkeypatch.setattr(storage_postgres, "add_followup", lambda _s, f: calls.append("add"))
    monkeypatch.setattr(storage_postgres, "list_open_followups", lambda _s: [row])
    monkeypatch.setattr(storage_postgres, "cancel_followup", lambda _s, i: calls.append(f"cancel:{i}") or True)
    monkeypatch.setattr(storage_postgres, "update_followup", lambda _s, i, d, t, c: calls.append(f"update:{i}") or True)
    monkeypatch.setattr(storage_postgres, "claim_due_followups", lambda _s, fired_at, c: [row])

    followups.add(settings, current + timedelta(hours=1), "ask about the viewing", "on Storgata")
    assert followups.list_open(settings) == [row]
    # cancel/update resolve the target via the (delegated) list_open, then flip it.
    assert followups.cancel(settings, "viewing").id == "f1"
    assert followups.update(settings, "f1", context="booked").id == "f1"
    assert [f.id for f in followups.claim_due(settings, current)] == ["f1"]
    assert calls == ["add", "cancel:f1", "update:f1"]


def test_postgres_threads_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant import storage_postgres, threads

    settings = _pg_settings()
    info = threads.ThreadInfo(
        thread_id="telegram:7", channel="telegram",
        last_user_at="2026-07-15T10:00:00+02:00", last_assistant_at="",
    )
    touched: list[tuple] = []
    monkeypatch.setattr(
        storage_postgres, "touch_thread",
        lambda _s, tid, ch, stamp, user, asst: touched.append((tid, ch, user, asst)),
    )
    monkeypatch.setattr(storage_postgres, "known_threads", lambda _s, channel=None: [info])

    threads.touch(settings, "telegram:7")
    assert threads.known_threads(settings) == [info]
    assert threads.last_contact(settings) is not None  # derived from known_threads
    assert touched == [("telegram:7", "telegram", True, True)]


def test_postgres_heartbeat_and_sleep_state_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant import heartbeat, sleep, storage_postgres

    settings = _pg_settings()
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(storage_postgres, "kv_get", lambda _s, ns, k: store.get((ns, k), ""))
    monkeypatch.setattr(storage_postgres, "kv_set", lambda _s, ns, k, v: store.__setitem__((ns, k), v))
    monkeypatch.setattr(storage_postgres, "kv_clear", lambda _s, ns, keys: [store.pop((ns, k), None) for k in keys])

    heartbeat.state_set(settings, "last_wake_at", "2026-07-15T09:00:00+02:00")
    assert heartbeat.state_get(settings, "last_wake_at") == "2026-07-15T09:00:00+02:00"
    heartbeat.state_clear(settings, "last_wake_at")
    assert heartbeat.state_get(settings, "last_wake_at") == ""
    assert ("heartbeat", "last_wake_at") not in store

    sleep._state_set(settings, "last_llm_pass_at", "2026-07-15")
    assert sleep._state_get(settings, "last_llm_pass_at") == "2026-07-15"
    assert store[("sleep", "last_llm_pass_at")] == "2026-07-15"  # separate namespace


def test_postgres_fired_ledger_claim_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant import briefing, fired_ledger, storage_postgres
    from assistant.calendar.context import now

    settings = _pg_settings()
    seen: list[tuple] = []

    def fake_claim(_s, ledger, keys, fired_at, current):
        seen.append((ledger, keys))
        return [0]  # first key newly claimed

    monkeypatch.setattr(storage_postgres, "claim_fired", fake_claim)
    current = now(settings)
    claimed = fired_ledger.claim(
        briefing._LEDGER, settings, [("2026-07-15",)], current.isoformat(), current
    )
    assert claimed == [0]
    assert seen == [("briefings_fired", [("2026-07-15",)])]


def test_postgres_mail_snapshot_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant import storage_postgres
    from assistant.calendar.context import now
    from assistant.mail import snapshot

    settings = _pg_settings().model_copy(update={"enable_email": True})
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(storage_postgres, "kv_get", lambda _s, ns, k: store.get((ns, k), ""))
    monkeypatch.setattr(storage_postgres, "kv_set", lambda _s, ns, k, v: store.__setitem__((ns, k), v))

    snapshot._save(settings, "1 unread message(s):\n- Hei", now(settings))
    assert ("mail", "snapshot") in store  # persisted to KV, not a file
    loaded = snapshot._load(settings)
    assert loaded is not None and loaded[0].startswith("1 unread message(s)")
