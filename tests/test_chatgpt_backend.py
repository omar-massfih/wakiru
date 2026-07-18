"""Tests for the chatgpt.com backend — no real network.

All HTTP goes through ``urllib.request.urlopen``, monkeypatched here with fakes
that dispatch on the request URL (token endpoint vs. responses endpoint), the
same style as test_agent.py's monkeypatched subprocess.
"""

from __future__ import annotations

import base64
import io
import json
import stat
import time
import urllib.error

import pytest

from assistant import chatgpt_backend
from assistant.chatgpt_backend import (
    ChatGptAuthError,
    ChatGptError,
    ChatGptTimeoutError,
    access_token,
    build_headers,
    build_payload,
    run_chatgpt,
    run_chatgpt_stream,
)
from assistant.config import Settings


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"header.{payload.decode()}.signature"


def _fresh_token(account: str | None = "acct-1") -> str:
    claims: dict = {"exp": time.time() + 3600}
    if account:
        claims["https://api.openai.com/auth"] = {"chatgpt_account_id": account}
    return _jwt(claims)


def _write_auth(tmp_path, access: str, account_id: str | None = "acct-1"):
    """A fake auth.json; returns Settings pointed at it."""
    path = tmp_path / "auth.json"
    tokens = {"access_token": access, "refresh_token": "rt-old", "id_token": "id-old"}
    if account_id:
        tokens["account_id"] = account_id
    path.write_text(json.dumps({"tokens": tokens, "last_refresh": "2026-01-01"}))
    return path, Settings(chatgpt_auth_file=str(path))


class _FakeResponse:
    """Stands in for urlopen's return: iterable SSE lines + read() for JSON."""

    def __init__(self, lines: list[bytes] | None = None, body: bytes = b"{}"):
        self._lines = lines or []
        self._body = body
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _sse(*events: tuple[str, dict]) -> list[bytes]:
    lines: list[bytes] = []
    for etype, data in events:
        lines += [
            f"event: {etype}\n".encode(),
            f"data: {json.dumps(data)}\n".encode(),
            b"\n",
        ]
    return lines


def _reply_events(*deltas: str) -> list[bytes]:
    events = [("response.output_text.delta", {"delta": d}) for d in deltas]
    events.append(("response.completed", {"response": {}}))
    return _sse(*events)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        chatgpt_backend._RESPONSES_URL, code, "nope", None, io.BytesIO(b"denied")
    )


# --- request building --------------------------------------------------------- #


def test_build_payload_shape() -> None:
    payload = build_payload("hello there", Settings(chatgpt_model="gpt-5"))
    assert payload["model"] == "gpt-5"
    assert payload["stream"] is True and payload["store"] is False
    [item] = payload["input"]
    assert item["role"] == "user"
    assert item["content"] == [{"type": "input_text", "text": "hello there"}]


def test_build_headers_shape() -> None:
    headers = build_headers("tok", "acct", "sess")
    assert headers["Authorization"] == "Bearer tok"
    assert headers["chatgpt-account-id"] == "acct"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["session_id"] == "sess"
    assert headers["Accept"] == "text/event-stream"


# --- auth load / refresh ------------------------------------------------------ #


def test_access_token_fresh_needs_no_refresh(tmp_path, monkeypatch) -> None:
    token = _fresh_token()
    _, settings = _write_auth(tmp_path, token)

    def no_network(*args, **kwargs):
        raise AssertionError("fresh token must not hit the network")

    monkeypatch.setattr(chatgpt_backend.urllib.request, "urlopen", no_network)
    assert access_token(settings) == (token, "acct-1")


def test_account_id_falls_back_to_jwt_claim(tmp_path, monkeypatch) -> None:
    token = _fresh_token(account="acct-from-jwt")
    _, settings = _write_auth(tmp_path, token, account_id=None)
    monkeypatch.setattr(
        chatgpt_backend.urllib.request, "urlopen", lambda *a, **k: None
    )
    assert access_token(settings) == (token, "acct-from-jwt")


def test_expired_token_is_refreshed_and_persisted(tmp_path, monkeypatch) -> None:
    path, settings = _write_auth(tmp_path, _jwt({"exp": time.time() - 10}))
    new_token = _fresh_token()
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return _FakeResponse(
            body=json.dumps(
                {"access_token": new_token, "refresh_token": "rt-new", "id_token": "id-new"}
            ).encode()
        )

    monkeypatch.setattr(chatgpt_backend.urllib.request, "urlopen", fake_urlopen)
    assert access_token(settings) == (new_token, "acct-1")

    assert seen["url"] == chatgpt_backend._TOKEN_URL
    assert seen["body"]["grant_type"] == "refresh_token"
    assert seen["body"]["refresh_token"] == "rt-old"

    stored = json.loads(path.read_text())
    assert stored["tokens"]["access_token"] == new_token
    assert stored["tokens"]["refresh_token"] == "rt-new"  # rotated
    assert stored["last_refresh"] != "2026-01-01"
    # The rewritten file must stay private (it grants account access).
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_undecodable_jwt_counts_as_expired(tmp_path, monkeypatch) -> None:
    _, settings = _write_auth(tmp_path, "not-a-jwt")
    monkeypatch.setattr(
        chatgpt_backend.urllib.request,
        "urlopen",
        lambda request, timeout=None: _FakeResponse(
            body=json.dumps({"access_token": _fresh_token()}).encode()
        ),
    )
    token, _ = access_token(settings)
    assert token != "not-a-jwt"  # a refresh happened


def test_refresh_failure_raises_auth_error(tmp_path, monkeypatch) -> None:
    _, settings = _write_auth(tmp_path, _jwt({"exp": 0}))

    def failing(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(chatgpt_backend.urllib.request, "urlopen", failing)
    with pytest.raises(ChatGptAuthError, match="codex login"):
        access_token(settings)


def test_missing_auth_file_raises_auth_error(tmp_path) -> None:
    settings = Settings(chatgpt_auth_file=str(tmp_path / "nope.json"))
    with pytest.raises(ChatGptAuthError, match="codex login"):
        access_token(settings)


def test_auth_file_without_tokens_raises_auth_error(tmp_path) -> None:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"OPENAI_API_KEY": "sk-...", "tokens": {}}))
    with pytest.raises(ChatGptAuthError, match="codex login"):
        access_token(Settings(chatgpt_auth_file=str(path)))


# --- runners ------------------------------------------------------------------ #


def _stream_settings(tmp_path) -> Settings:
    _, settings = _write_auth(tmp_path, _fresh_token())
    return settings


def test_run_chatgpt_concatenates_deltas(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    response = _FakeResponse(_reply_events("Hel", "lo", " world"))
    monkeypatch.setattr(
        chatgpt_backend.urllib.request, "urlopen", lambda req, timeout=None: response
    )
    assert run_chatgpt("hi", settings=settings) == "Hello world"
    assert response.closed


def test_run_chatgpt_stream_yields_increments(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    monkeypatch.setattr(
        chatgpt_backend.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(_reply_events("a", "b")),
    )
    assert list(run_chatgpt_stream("hi", settings=settings)) == ["a", "b"]


def test_run_chatgpt_sends_prompt_and_auth(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["auth"] = request.get_header("Authorization")
        seen["account"] = request.get_header("Chatgpt-account-id")
        return _FakeResponse(_reply_events("ok"))

    monkeypatch.setattr(chatgpt_backend.urllib.request, "urlopen", fake_urlopen)
    long_prompt = "x" * 500_000  # streams fine — no argv limits here
    assert run_chatgpt(long_prompt, settings=settings) == "ok"
    assert seen["url"] == chatgpt_backend._RESPONSES_URL
    assert seen["body"]["input"][0]["content"][0]["text"] == long_prompt
    assert seen["auth"].startswith("Bearer ")
    assert seen["account"] == "acct-1"


def test_response_failed_event_raises(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    events = _sse(("response.failed", {"response": {"error": {"message": "usage limit hit"}}}))
    monkeypatch.setattr(
        chatgpt_backend.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(events),
    )
    with pytest.raises(ChatGptError, match="usage limit hit"):
        run_chatgpt("hi", settings=settings)


def test_empty_stream_raises(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    monkeypatch.setattr(
        chatgpt_backend.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse([]),
    )
    with pytest.raises(ChatGptError, match="without any reply"):
        run_chatgpt("hi", settings=settings)


def test_http_401_refreshes_once_and_retries(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    calls = {"responses": 0, "token": 0}

    def fake_urlopen(request, timeout=None):
        if request.full_url == chatgpt_backend._TOKEN_URL:
            calls["token"] += 1
            return _FakeResponse(
                body=json.dumps({"access_token": _fresh_token()}).encode()
            )
        calls["responses"] += 1
        if calls["responses"] == 1:
            raise _http_error(401)
        return _FakeResponse(_reply_events("recovered"))

    monkeypatch.setattr(chatgpt_backend.urllib.request, "urlopen", fake_urlopen)
    assert run_chatgpt("hi", settings=settings) == "recovered"
    assert calls == {"responses": 2, "token": 1}


def test_persistent_401_raises_chatgpt_error(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)

    def fake_urlopen(request, timeout=None):
        if request.full_url == chatgpt_backend._TOKEN_URL:
            return _FakeResponse(
                body=json.dumps({"access_token": _fresh_token()}).encode()
            )
        raise _http_error(401)

    monkeypatch.setattr(chatgpt_backend.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ChatGptError, match="HTTP 401"):
        run_chatgpt("hi", settings=settings)


def test_http_429_raises_chatgpt_error(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    monkeypatch.setattr(
        chatgpt_backend.urllib.request,
        "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(_http_error(429)),
    )
    with pytest.raises(ChatGptError, match="HTTP 429"):
        run_chatgpt("hi", settings=settings)


def test_connect_timeout_raises_timeout_error(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)

    def timing_out(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(chatgpt_backend.urllib.request, "urlopen", timing_out)
    with pytest.raises(ChatGptTimeoutError):
        run_chatgpt("hi", settings=settings)


def test_mid_stream_timeout_raises_timeout_error(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)

    class _StallingResponse(_FakeResponse):
        def __iter__(self):
            yield from _sse(("response.output_text.delta", {"delta": "par"}))
            raise TimeoutError("read timed out")

    monkeypatch.setattr(
        chatgpt_backend.urllib.request,
        "urlopen",
        lambda req, timeout=None: _StallingResponse(),
    )
    stream = run_chatgpt_stream("hi", settings=settings)
    assert next(stream) == "par"
    with pytest.raises(ChatGptTimeoutError):
        list(stream)


def test_closing_stream_early_closes_response(tmp_path, monkeypatch) -> None:
    settings = _stream_settings(tmp_path)
    response = _FakeResponse(_reply_events("a", "b", "c"))
    monkeypatch.setattr(
        chatgpt_backend.urllib.request, "urlopen", lambda req, timeout=None: response
    )
    stream = run_chatgpt_stream("hi", settings=settings)
    assert next(stream) == "a"
    stream.close()
    assert response.closed
