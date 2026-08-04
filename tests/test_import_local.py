from __future__ import annotations

from assistant import import_local
from assistant.config import Settings
from assistant.tasks.store import Task, TaskCompletion


def test_import_tasks_restores_history_before_terminal_tasks(monkeypatch) -> None:
    local = Settings(storage_backend="local")
    postgres = Settings(
        storage_backend="postgres", database_url="postgres://example"
    )
    task = Task(id="t1", title="Pay bill", done=True, done_at="completed-at")
    completion = TaskCompletion(
        id="real-occurrence", task_id=task.id, title=task.title,
        completed_at=task.done_at,
    )
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        import_local.task_store, "list_tasks", lambda _settings, include_done: [task]
    )
    monkeypatch.setattr(
        import_local.task_store, "list_task_completions",
        lambda _settings, include_undone: [completion],
    )
    monkeypatch.setattr(
        import_local.storage_postgres, "restore_task_completion",
        lambda _settings, value: calls.append(("completion", value.id)),
    )
    monkeypatch.setattr(
        import_local.storage_postgres, "restore_task",
        lambda _settings, value: calls.append(("task", value.id)),
    )

    assert import_local.import_tasks(local, postgres) == 1
    assert calls == [("completion", "real-occurrence"), ("task", "t1")]


def test_restore_completion_reconciles_legacy_backfill_on_rerun(monkeypatch) -> None:
    """Schema setup may backfill an old terminal destination before import."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from assistant.storage_postgres import tasks as pg_tasks

    active = {"legacy:t1"}

    class Cursor:
        def __init__(self, rows=()):
            self.description = [SimpleNamespace(name="occurrence_seq")]
            self._rows = list(rows)

        def fetchall(self):
            return self._rows

    class FakeConn:
        def execute(self, sql, params=()):
            if "WHERE id = %s" in sql and sql.startswith("SELECT occurrence_seq"):
                return Cursor(((1,),)) if params[0] in active else Cursor()
            if "MAX(occurrence_seq)" in sql:
                return Cursor(((1,),))
            if sql.startswith("DELETE FROM assistant_task_completions"):
                active.discard(params[0])
            if "INSERT INTO assistant_task_completions" in sql:
                active.add(params[0])
            return Cursor()

    @contextmanager
    def fake_connect(_settings):
        yield FakeConn()

    # Represents a destination created by an older import: terminal task exists,
    # completion history does not, and schema setup has synthesized legacy:t1.
    monkeypatch.setattr(pg_tasks, "ensure_tasks_schema", lambda _settings: None)
    monkeypatch.setattr(pg_tasks, "connect", fake_connect)
    completion = TaskCompletion(
        id="real-occurrence", task_id="t1", title="Pay bill",
        completed_at="2026-07-01T12:00:00+00:00",
    )

    pg_tasks.restore_task_completion(Settings(), completion)
    pg_tasks.restore_task_completion(Settings(), completion)  # rerun remains idempotent

    assert active == {"real-occurrence"}
