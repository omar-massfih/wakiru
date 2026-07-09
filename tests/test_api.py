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


def test_tasks_shape_lists_open_and_all(client) -> None:
    import assistant.api as api
    from assistant.tasks import store as tstore

    c = client(None)
    app_settings = api.get_settings()
    keep = tstore.create_task(app_settings, "call plumber")
    done = tstore.create_task(app_settings, "buy milk")
    tstore.complete_task(app_settings, done.id)

    # Default: open tasks only.
    body = c.get("/tasks").json()
    assert body["total"] == 1
    assert body["tasks"][0]["title"] == "call plumber"
    assert body["tasks"][0]["done"] is False
    assert {"id", "title", "done", "due", "notes", "done_at"} <= body["tasks"][0].keys()

    # include_done surfaces the completed one too.
    all_body = c.get("/tasks", params={"include_done": "true"}).json()
    assert all_body["total"] == 2
    assert any(t["done"] for t in all_body["tasks"])
    _ = keep


def test_docs_ingest_list_search_and_delete(client, monkeypatch) -> None:
    import math
    import re
    import zlib

    def _fake_embed(texts, prefix="", settings=None):
        vecs = []
        for text in texts:
            v = [0.0] * 64
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[zlib.crc32(word.encode()) % 64] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs

    # Keep the real sqlite-vec index but skip the 2GB embedding model.
    monkeypatch.setattr("assistant.memory.embeddings._embed", _fake_embed)

    c = client(None)
    created = c.post("/documents", json={"title": "Bergen", "text": "The fish market sells salmon."})
    assert created.status_code == 200
    doc_id = created.json()["id"]
    assert created.json()["chunks"] == 1

    listed = c.get("/documents").json()
    assert listed["total"] == 1 and listed["documents"][0]["title"] == "Bergen"

    found = c.get("/documents/search", params={"q": "salmon fish market"}).json()
    assert found["total"] >= 1 and found["chunks"][0]["doc_title"] == "Bergen"

    assert c.delete(f"/documents/{doc_id}").json()["deleted"] is True
    assert c.get("/documents").json()["total"] == 0
    assert c.delete(f"/documents/{doc_id}").status_code == 404


def test_email_endpoints_409_while_disabled(client) -> None:
    # Email is off by default, so every mail endpoint refuses before touching a socket.
    c = client(None)
    assert c.get("/email").status_code == 409
    assert c.get("/email/1").status_code == 409
    resp = c.post("/email/draft", json={"to": "b@x.com", "subject": "s", "body": "b"})
    assert resp.status_code == 409
    assert "disabled" in resp.json()["detail"].lower()


def test_email_draft_does_not_send_by_default(client, monkeypatch) -> None:
    import assistant.api as api

    calls: list[str] = []
    monkeypatch.setattr(
        api.mail_client, "save_draft", lambda s, to, subj, body: calls.append("draft") or "drafted: x"
    )
    monkeypatch.setattr(
        api.mail_client, "send_message", lambda *a, **k: pytest.fail("must not send")
    )
    body = client(None).post(
        "/email/draft", json={"to": "b@x.com", "subject": "s", "body": "b"}
    ).json()
    assert body["sent"] is False
    assert calls == ["draft"]


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


# --- slack route + web UI --------------------------------------------------- #


def test_slack_route_404_when_unconfigured(client) -> None:
    assert client(None).post("/slack/events", json={"type": "x"}).status_code == 404


def test_slack_route_rejects_bad_signature(tmp_path, monkeypatch) -> None:
    import assistant.api as api

    settings = Settings(
        memory_dir=str(tmp_path / "m"),
        slack_bot_token="xoxb",
        slack_signing_secret="secret",
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    c = TestClient(api.app)
    resp = c.post(
        "/slack/events",
        content=b"{}",
        headers={"x-slack-request-timestamp": "1", "x-slack-signature": "v0=bad"},
    )
    assert resp.status_code == 401


def test_slack_route_answers_url_verification(tmp_path, monkeypatch) -> None:
    import hashlib
    import hmac
    import json
    import time

    import assistant.api as api

    settings = Settings(
        memory_dir=str(tmp_path / "m"),
        slack_bot_token="xoxb",
        slack_signing_secret="secret",
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)

    body = json.dumps({"type": "url_verification", "challenge": "abc123"}).encode()
    ts = str(int(time.time()))
    sig = "v0=" + hmac.new(
        b"secret", b"v0:" + ts.encode() + b":" + body, hashlib.sha256
    ).hexdigest()

    resp = TestClient(api.app).post(
        "/slack/events",
        content=body,
        headers={"x-slack-request-timestamp": ts, "x-slack-signature": sig},
    )
    assert resp.status_code == 200
    assert resp.json() == {"challenge": "abc123"}


def test_ui_serves_html_without_token(client) -> None:
    # The page carries no data, so it is not token-gated even when API_TOKEN is set.
    resp = client("sekrit").get("/ui")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "/chat/stream" in resp.text


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
