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


def test_heartbeat_run_noop_when_disabled(client, monkeypatch) -> None:
    # Heartbeat is off by default: the triage skips before any model wiring,
    # so the endpoint answers without an agent or LLM in play.
    monkeypatch.setattr("assistant.api._agent", lambda: None)
    body = client(None).post("/heartbeat/run").json()
    assert body == {"sent": False, "reason": "nothing to do"}


def test_heartbeat_run_requires_token_when_set(client, monkeypatch) -> None:
    monkeypatch.setattr("assistant.api._agent", lambda: None)
    assert client("sekrit").post("/heartbeat/run").status_code == 401


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


def test_slack_route_acks_a_retry_without_running_the_turn(tmp_path, monkeypatch) -> None:
    import hashlib
    import hmac
    import json
    import time

    import assistant.api as api
    from assistant import slack

    settings = Settings(
        memory_dir=str(tmp_path / "m"),
        slack_bot_token="xoxb",
        slack_signing_secret="secret",
        slack_allowed_user_ids=["U1"],
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(
        slack, "handle_event", lambda *a, **k: pytest.fail("a retry must not run the turn")
    )

    body = json.dumps(
        {"event_id": "Ev-retry", "event": {"type": "message", "user": "U1", "text": "hi"}}
    ).encode()
    ts = str(int(time.time()))
    sig = "v0=" + hmac.new(
        b"secret", b"v0:" + ts.encode() + b":" + body, hashlib.sha256
    ).hexdigest()

    resp = TestClient(api.app).post(
        "/slack/events",
        content=body,
        headers={
            "x-slack-request-timestamp": ts,
            "x-slack-signature": sig,
            "x-slack-retry-num": "1",
        },
    )
    assert resp.status_code == 200 and resp.json() == {"ok": True}


def test_slack_retry_header_is_ignored_without_a_valid_signature(tmp_path, monkeypatch) -> None:
    import assistant.api as api

    settings = Settings(
        memory_dir=str(tmp_path / "m"),
        slack_bot_token="xoxb",
        slack_signing_secret="secret",
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)

    # An unverified header proves nothing: the 401 must still win.
    resp = TestClient(api.app).post(
        "/slack/events",
        content=b"{}",
        headers={
            "x-slack-request-timestamp": "1",
            "x-slack-signature": "v0=bad",
            "x-slack-retry-num": "1",
        },
    )
    assert resp.status_code == 401


def test_ui_serves_html_without_token(client) -> None:
    # The page carries no data, so it is not token-gated even when API_TOKEN is set.
    resp = client("sekrit").get("/ui")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "/chat/stream" in resp.text


# --- streaming endpoint --------------------------------------------------- #


def _parse_sse(body: str) -> tuple[str, dict[str, str]]:
    """Reconstruct an SSE stream the way a spec-compliant client (webui.py) does:
    a frame ends at a blank line, and its payload is the ``data:`` lines joined
    by newlines. Returns (concatenated data payloads, {event_name: payload})."""
    text, events = "", {}
    for frame in body.split("\n\n"):
        if not frame:
            continue
        name, data = None, []
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        payload = "\n".join(data)
        if name:
            events[name] = payload
        else:
            text += payload
    return text, events


def test_chat_stream_preserves_newlines_and_blank_lines(client, monkeypatch) -> None:
    """A reply with a paragraph break must survive the wire intact.

    Regression: encoding a chunk as a single ``data: {chunk}`` line meant a blank
    line inside it terminated the frame early, silently truncating the reply.
    The Codex provider yields the whole reply as one chunk, so this hit nearly
    every multi-paragraph answer.
    """
    import assistant.api as api

    reply = "Line one.\nLine two.\n\nNew paragraph."

    async def fake_stream(agent, message, thread_id, settings=None):
        yield reply  # one chunk, as the codex provider does

    monkeypatch.setattr(api, "run_chat_stream", fake_stream)
    monkeypatch.setattr(api, "_agent", lambda: object())
    monkeypatch.setattr(api, "run_upkeep", lambda *a, **k: None)

    resp = client(None).post("/chat/stream", json={"message": "hi", "thread_id": "t1"})
    text, events = _parse_sse(resp.text)
    assert text == reply  # lossless, blank line and all
    assert events["done"] == "t1"


def test_chat_stream_error_frame_survives_multiline_message(client, monkeypatch) -> None:
    import assistant.api as api
    from assistant.codex_runner import CodexError

    async def boom(agent, message, thread_id, settings=None):
        raise CodexError("failed:\n\nstderr detail")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(api, "run_chat_stream", boom)
    monkeypatch.setattr(api, "_agent", lambda: object())

    resp = client(None).post("/chat/stream", json={"message": "hi"})
    _, events = _parse_sse(resp.text)
    assert events["error"] == "failed:\n\nstderr detail"


def test_sse_frame_encodes_one_data_line_per_content_line() -> None:
    from assistant.api import sse_frame

    assert sse_frame("a\n\nb") == "data: a\ndata: \ndata: b\n\n"
    assert sse_frame("x", event="done") == "event: done\ndata: x\n\n"


def test_chat_stream_runs_upkeep_once_after_the_stream(client, monkeypatch) -> None:
    """The background task is added from *inside* the generator, after the
    response object was returned — assert it still runs, with the full reply."""
    import assistant.api as api

    async def fake_stream(agent, message, thread_id, settings=None):
        yield "hel"
        yield "lo"

    ran: list[tuple] = []
    monkeypatch.setattr(api, "run_chat_stream", fake_stream)
    monkeypatch.setattr(api, "_agent", lambda: object())
    monkeypatch.setattr(api, "run_upkeep", lambda a, s, m, r, t: ran.append((m, r, t)))

    client(None).post("/chat/stream", json={"message": "hi", "thread_id": "t1"})
    assert ran == [("hi", "hello", "t1")]  # once, with the reassembled reply


def test_chat_stream_skips_upkeep_when_the_model_fails(client, monkeypatch) -> None:
    import assistant.api as api
    from assistant.codex_runner import CodexError

    async def boom(agent, message, thread_id, settings=None):
        raise CodexError("nope")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(api, "run_chat_stream", boom)
    monkeypatch.setattr(api, "_agent", lambda: object())
    monkeypatch.setattr(api, "run_upkeep", lambda *a: pytest.fail("no upkeep on failure"))
    client(None).post("/chat/stream", json={"message": "hi"})


def test_slack_route_rejects_malformed_json(tmp_path, monkeypatch) -> None:
    import hashlib
    import hmac
    import time

    import assistant.api as api

    settings = Settings(
        memory_dir=str(tmp_path / "m"),
        slack_bot_token="xoxb",
        slack_signing_secret="secret",
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)

    body = b"not json"
    ts = str(int(time.time()))
    sig = "v0=" + hmac.new(
        b"secret", b"v0:" + ts.encode() + b":" + body, hashlib.sha256
    ).hexdigest()
    resp = TestClient(api.app).post(
        "/slack/events",
        content=body,
        headers={"x-slack-request-timestamp": ts, "x-slack-signature": sig},
    )
    # 400, not 500 — a 5xx would make Slack retry the same bad payload.
    assert resp.status_code == 400


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


# --- startup checks (run inside lifespan, i.e. only under `with TestClient`) -- #


@pytest.fixture
def startup(tmp_path, monkeypatch):
    """Build settings for lifespan tests; reminders off so no background tasks start."""

    def make(**overrides):
        settings = Settings(
            memory_dir=str(tmp_path / "memory"), enable_reminders=False, **overrides
        )
        monkeypatch.setattr("assistant.api.get_settings", lambda: settings)
        return TestClient(app)

    return make


def test_startup_refuses_non_loopback_bind_without_token(startup) -> None:
    with pytest.raises(RuntimeError, match="API_TOKEN"), startup(host="0.0.0.0"):
        pass


def test_allow_unauthenticated_overrides_the_refusal(startup, caplog) -> None:
    with (
        caplog.at_level("WARNING", logger="assistant.api"),
        startup(host="0.0.0.0", allow_unauthenticated=True),
    ):
        pass
    assert any("ALLOW_UNAUTHENTICATED" in r.message for r in caplog.records)


def test_loopback_bind_without_token_starts_silently(startup, caplog) -> None:
    with caplog.at_level("WARNING", logger="assistant.api"), startup(host="127.0.0.1"):
        pass
    assert not caplog.records


def test_startup_warns_on_writable_sandbox_with_remote_channel(startup, caplog) -> None:
    # Slack (not Telegram) so lifespan starts no polling task.
    with (
        caplog.at_level("WARNING", logger="assistant.api"),
        startup(codex_sandbox="workspace-write", slack_bot_token="xoxb-test"),
    ):
        pass
    assert any("CODEX_SANDBOX" in r.message for r in caplog.records)


def test_writable_sandbox_alone_on_loopback_does_not_warn(startup, caplog) -> None:
    with (
        caplog.at_level("WARNING", logger="assistant.api"),
        startup(codex_sandbox="workspace-write"),
    ):
        pass
    assert not caplog.records


# --- request size limits ---------------------------------------------------- #


def test_oversized_chat_message_is_rejected(client) -> None:
    resp = client(None).post("/chat", json={"message": "x" * 100_001})
    assert resp.status_code == 422


def test_oversized_doc_text_is_rejected(client) -> None:
    resp = client(None).post(
        "/documents", json={"title": "big", "text": "x" * 2_000_001}
    )
    assert resp.status_code == 422


def test_doc_under_the_limit_ingests(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )
    resp = client(None).post("/documents", json={"title": "ok", "text": "short doc"})
    assert resp.status_code == 200


def test_oversized_search_query_is_rejected(client) -> None:
    resp = client(None).get("/documents/search", params={"q": "x" * 1_001})
    assert resp.status_code == 422


def test_oversized_upload_is_rejected(tmp_path, monkeypatch) -> None:
    import assistant.api as api

    settings = Settings(memory_dir=str(tmp_path / "m"), docs_upload_max_bytes=1_000)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    resp = TestClient(api.app).post(
        "/documents/upload", files={"file": ("big.txt", b"x" * 1_001, "text/plain")}
    )
    assert resp.status_code == 413


def test_upload_under_the_limit_ingests(tmp_path, monkeypatch) -> None:
    import assistant.api as api

    settings = Settings(memory_dir=str(tmp_path / "m"), docs_upload_max_bytes=1_000)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )
    resp = TestClient(api.app).post(
        "/documents/upload", files={"file": ("ok.txt", b"short doc", "text/plain")}
    )
    assert resp.status_code == 200
