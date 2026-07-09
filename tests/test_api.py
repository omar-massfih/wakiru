"""API tests — bearer-token gating plus the read endpoints' response shapes.

Only the cheap endpoints are exercised (``/memory``, ``/calendar``,
``/reminders/run``, ``/memory/consolidate`` over an empty store), so no Codex,
network, or agent build is involved. ``/chat`` is covered via the chat core in
``test_chat.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assistant.api import app
from assistant.config import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    def make(api_token: str | None):
        settings = Settings(memory_dir=str(tmp_path / "memory"), api_token=api_token)
        monkeypatch.setattr("assistant.api.get_settings", lambda: settings)
        return TestClient(app)

    return make


def test_health_stays_open_with_token_set(client) -> None:
    assert client("sekrit").get("/health").status_code == 200


def test_endpoints_reject_missing_or_wrong_token(client) -> None:
    c = client("sekrit")
    assert c.get("/memory").status_code == 401
    assert c.get("/memory", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/calendar", headers={"Authorization": "sekrit"}).status_code == 401  # not Bearer
    assert c.post("/reminders/run").status_code == 401


def test_endpoints_accept_valid_token(client) -> None:
    c = client("sekrit")
    headers = {"Authorization": "Bearer sekrit"}
    assert c.get("/memory", headers=headers).status_code == 200
    assert c.get("/calendar", headers=headers).status_code == 200


def test_unset_token_keeps_legacy_open_behavior(client) -> None:
    c = client(None)
    assert c.get("/memory").status_code == 200
    assert c.get("/calendar").status_code == 200


# --- response shapes (endpoint bodies, not just the auth gate) ------------- #


def test_memory_shape_on_empty_store(client) -> None:
    body = client(None).get("/memory").json()
    assert body["total"] == 0
    assert body["by_kind"] == {}
    assert body["notes"] == []


def test_calendar_shape_lists_events(client) -> None:
    from datetime import timedelta

    import assistant.api as api
    from assistant.calendar import context, store

    c = client(None)
    # The endpoint reads through the monkeypatched get_settings; create an event
    # against that same store, within the upcoming horizon, then assert it surfaces.
    app_settings = api.get_settings()
    start = (context.now(app_settings) + timedelta(days=1)).isoformat(timespec="seconds")
    store.create_event(app_settings, title="Dentist", start=start)

    body = c.get("/calendar").json()
    assert "now" in body
    assert body["total"] == 1
    event = body["events"][0]
    assert event["title"] == "Dentist"
    assert {"id", "title", "start", "end", "location", "notes"} <= event.keys()


def test_reminders_run_shape_on_empty_store(client) -> None:
    body = client(None).post("/reminders/run").json()
    assert body["count"] == 0
    assert body["fired"] == []


def test_memory_consolidate_shape_on_empty_store(client) -> None:
    # An empty brain has nothing to reconcile, so this returns a summary dict
    # without making any Codex call.
    resp = client(None).post("/memory/consolidate")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


# --- streaming endpoint --------------------------------------------------- #


def test_chat_stream_emits_sse_frames_and_done(client, monkeypatch) -> None:
    import assistant.api as api

    async def fake_stream(agent, message, thread_id, settings=None):
        for text in ("Hel", "lo"):
            yield text

    # Avoid building the real graph / making Codex calls; the post-stream upkeep
    # is exercised elsewhere (test_chat.py) and would load the embedding model.
    monkeypatch.setattr(api, "run_chat_stream", fake_stream)
    monkeypatch.setattr(api, "_agent", lambda: object())
    monkeypatch.setattr(api, "run_upkeep", lambda *a, **k: None)

    c = client(None)
    resp = c.post("/chat/stream", json={"message": "hi", "thread_id": "t1"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "data: Hel\n\n" in body
    assert "data: lo\n\n" in body
    # Terminal frame carries the thread id.
    assert "event: done\ndata: t1\n\n" in body
