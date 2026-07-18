"""Chat via chatgpt.com's backend Responses endpoint, stdlib-only.

Reuses the ChatGPT OAuth tokens the Codex CLI keeps in ``$CODEX_HOME/auth.json``
(``codex login``), so there is no API key and usage is billed to the ChatGPT
plan. The endpoint is the same private one the Codex CLI itself drives —
unofficial, so everything that touches it lives in this one module.

Public shape mirrors :mod:`assistant.codex_runner`: ``run_chatgpt`` /
``run_chatgpt_stream`` with the same signatures and error semantics, so
``llm.py`` can swap runners and nothing upstream changes.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://auth.openai.com/oauth/token"
# The Codex CLI's public OAuth client id (baked into the open-source CLI).
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
# The HTTP-SSE contract. Newer codex builds have moved to a websocket variant
# (responses_websockets=...); if chatgpt.com ever drops SSE, this is the knob.
_OPENAI_BETA = "responses=experimental"
# The endpoint requires an instructions field; the real system content rides in
# the rendered prompt (llm._render_prompt), so this stays minimal and stable.
_INSTRUCTIONS = "You are a helpful assistant."
_REFRESH_TIMEOUT_SECONDS = 30
# Refresh a little early so a token can't expire mid-request.
_EXPIRY_SKEW_SECONDS = 300

# Guards read-modify-write of auth.json across this process's threads (a chat
# turn fans out into parallel upkeep calls; refresh tokens are single-use).
_auth_lock = threading.Lock()

# Cap on concurrent requests (see run_chatgpt) — rate limits here are the
# ChatGPT plan's, and one chat turn fans out into several calls.
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_lock = threading.Lock()


def _chatgpt_slot(settings: Settings) -> threading.BoundedSemaphore:
    global _semaphore
    with _semaphore_lock:
        if _semaphore is None:
            _semaphore = threading.BoundedSemaphore(
                max(settings.chatgpt_max_concurrency, 1)
            )
        return _semaphore


class ChatGptError(RuntimeError):
    """Raised when the chatgpt.com backend request fails."""


class ChatGptTimeoutError(ChatGptError):
    """Raised when a request exceeds ``chatgpt_timeout``.

    A subclass so ``except ChatGptError`` callers keep working, while channels
    can tell "took too long" from "broke" when explaining a failure.
    """


class ChatGptAuthError(ChatGptError):
    """Raised when auth.json is missing/broken or the token refresh fails.

    Distinct from :class:`ChatGptError` because the fix is user-actionable:
    re-run ``codex login``.
    """


# --------------------------------------------------------------------------- #
# Auth: load / refresh / persist the Codex CLI's OAuth tokens
# --------------------------------------------------------------------------- #


def _auth_path(settings: Settings) -> Path:
    if settings.chatgpt_auth_file:
        return Path(settings.chatgpt_auth_file).expanduser()
    home = os.environ.get("CODEX_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".codex"
    return root / "auth.json"


def _load_auth(path: Path) -> dict:
    try:
        auth = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ChatGptAuthError(
            f"No Codex auth file at {path} — run `codex login` first."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ChatGptAuthError(f"Could not read Codex auth file {path}: {exc}") from exc
    tokens = auth.get("tokens") or {}
    if not tokens.get("access_token") or not tokens.get("refresh_token"):
        raise ChatGptAuthError(
            f"Codex auth file {path} has no ChatGPT tokens — run `codex login` "
            "(API-key-only auth cannot reach the chatgpt.com backend)."
        )
    return auth


def _jwt_claims(token: str) -> dict:
    """The JWT's payload claims; ``{}`` when the token doesn't decode."""
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (IndexError, ValueError, binascii.Error):
        return {}
    return claims if isinstance(claims, dict) else {}


def _jwt_expiry(token: str) -> float:
    """The access token's ``exp`` (epoch seconds); 0.0 => treat as expired."""
    exp = _jwt_claims(token).get("exp")
    return float(exp) if isinstance(exp, int | float) else 0.0


def _account_id(auth: dict) -> str:
    tokens = auth.get("tokens") or {}
    account = tokens.get("account_id")
    if account:
        return str(account)
    claims = _jwt_claims(tokens.get("access_token") or "")
    account = (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
    if account:
        return str(account)
    raise ChatGptAuthError(
        "Could not determine the ChatGPT account id from auth.json — "
        "run `codex login` to refresh it."
    )


def _store_auth(path: Path, auth: dict) -> None:
    # The tokens grant full account access, so the file must never be
    # loose-permissioned, even briefly: create 0600 from the start and publish
    # atomically (same idiom as mail/oauth._store_cached).
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(auth, indent=2))
    os.replace(tmp, path)


def _refresh(auth: dict, path: Path) -> dict:
    """Exchange the refresh token for fresh tokens and rewrite auth.json."""
    tokens = auth.get("tokens") or {}
    body = json.dumps(
        {
            "client_id": _CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": tokens.get("refresh_token"),
            "scope": "openid profile email",
        }
    ).encode()
    request = urllib.request.Request(
        _TOKEN_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_REFRESH_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read())
    except Exception as exc:  # urllib raises a zoo of errors; all mean "no token"
        raise ChatGptAuthError(
            f"Refreshing the ChatGPT token failed ({exc}) — try `codex login`."
        ) from exc

    access = data.get("access_token")
    if not access:
        raise ChatGptAuthError("Token endpoint returned no access_token.")
    tokens["access_token"] = access
    if data.get("refresh_token"):  # rotated — the old one is now dead
        tokens["refresh_token"] = data["refresh_token"]
    if data.get("id_token"):
        tokens["id_token"] = data["id_token"]
    auth["tokens"] = tokens
    auth["last_refresh"] = datetime.now(UTC).isoformat()
    try:
        _store_auth(path, auth)
    except OSError:
        # The fresh token still works for this process; losing the rotated
        # refresh token on disk is worth a loud warning though.
        logger.warning("could not write refreshed tokens to %s", path, exc_info=True)
    return auth


def access_token(
    settings: Settings, force_refresh: bool = False
) -> tuple[str, str]:
    """A valid ``(access_token, account_id)`` pair, refreshed when near expiry."""
    path = _auth_path(settings)
    with _auth_lock:
        auth = _load_auth(path)
        token = auth["tokens"]["access_token"]
        if force_refresh or _jwt_expiry(token) - time.time() < _EXPIRY_SKEW_SECONDS:
            auth = _refresh(auth, path)
            token = auth["tokens"]["access_token"]
        return token, _account_id(auth)


# --------------------------------------------------------------------------- #
# Request building (pure, unit-testable)
# --------------------------------------------------------------------------- #


def build_payload(prompt: str, settings: Settings) -> dict:
    """The Responses-API-shaped request body. Streaming is mandatory here."""
    return {
        "model": settings.chatgpt_model,
        "instructions": _INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "stream": True,
        "store": False,
    }


def build_headers(token: str, account_id: str, session_id: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": _OPENAI_BETA,
        "originator": "codex_cli_rs",
        "session_id": session_id,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


# --------------------------------------------------------------------------- #
# SSE parsing
# --------------------------------------------------------------------------- #


def iter_sse(lines: Iterable[bytes | str]) -> Iterator[tuple[str, dict]]:
    """Decode an SSE byte/line stream into ``(event_type, data)`` pairs.

    Frames are ``event:``/``data:`` lines terminated by a blank line; multi-line
    ``data:`` payloads are joined before JSON-decoding. ``[DONE]`` sentinels and
    non-JSON payloads are skipped. When a frame has no ``event:`` line the
    payload's own ``type`` field is used (the Responses API sets both).
    """
    event_type = ""
    data_lines: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if line == "":
            etype, payload = event_type, "\n".join(data_lines)
            event_type, data_lines = "", []
            if not payload or payload.strip() == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            if isinstance(data, dict):
                yield etype or str(data.get("type", "")), data
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())


class ChatGptStreamParser:
    """Reduce Responses SSE events to user-visible text deltas.

    Only ``response.output_text.delta`` carries reply text; reasoning-summary
    deltas and item bookkeeping are ignored. Failure events are collected on
    :attr:`failure` rather than raised, so the stream owner decides how to
    surface them after the stream (mirrors :class:`CodexStreamParser`).
    """

    def __init__(self) -> None:
        self.emitted = ""
        self.completed = False
        self.failure: str | None = None

    def feed(self, event_type: str, data: dict) -> list[str]:
        """The deltas one SSE event unlocks (often none)."""
        if event_type == "response.output_text.delta":
            delta = data.get("delta") or ""
            if not isinstance(delta, str) or not delta:
                return []
            self.emitted += delta
            return [delta]
        if event_type == "response.completed":
            self.completed = True
        elif event_type == "response.failed":
            error = (data.get("response") or {}).get("error") or data.get("error") or {}
            self.failure = error.get("message") or self.failure or "response.failed"
        elif event_type == "error":
            self.failure = self.failure or data.get("message") or "error event"
        return []


# --------------------------------------------------------------------------- #
# Runners — same shape as run_codex / run_codex_stream
# --------------------------------------------------------------------------- #


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")[:500]
    except OSError:
        body = ""
    return f"HTTP {exc.code}: {body or exc.reason}"


def _open_stream(prompt: str, settings: Settings):
    """POST the prompt and return the live SSE response (caller closes it).

    A 401 gets one forced token refresh and retry — the JWT can be revoked
    server-side before its ``exp``.
    """
    payload = json.dumps(build_payload(prompt, settings)).encode()
    for attempt in (0, 1):
        token, account = access_token(settings, force_refresh=attempt > 0)
        request = urllib.request.Request(
            _RESPONSES_URL,
            data=payload,
            method="POST",
            headers=build_headers(token, account, str(uuid.uuid4())),
        )
        try:
            return urllib.request.urlopen(request, timeout=settings.chatgpt_timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and attempt == 0:
                logger.info("chatgpt backend returned 401; refreshing token")
                continue
            raise ChatGptError(
                f"chatgpt.com request failed — {_http_error_detail(exc)}"
            ) from exc
        except TimeoutError as exc:
            raise ChatGptTimeoutError(
                f"chatgpt.com did not respond within {settings.chatgpt_timeout}s."
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ChatGptTimeoutError(
                    f"chatgpt.com did not respond within {settings.chatgpt_timeout}s."
                ) from exc
            raise ChatGptError(f"chatgpt.com request failed: {exc.reason}") from exc
    raise AssertionError("unreachable")


def run_chatgpt(prompt: str, settings: Settings | None = None) -> str:
    """Run one turn against the chatgpt.com backend and return the reply text.

    Concurrency is bounded by ``chatgpt_max_concurrency``: one chat turn fans
    out into several calls (reply, then memory/calendar/summary upkeep), and
    rate limits here are the ChatGPT plan's. Excess calls queue for a slot.
    """
    settings = settings or get_settings()
    return "".join(_stream_deltas(prompt, settings)).strip()


def run_chatgpt_stream(prompt: str, settings: Settings | None = None) -> Iterator[str]:
    """Run one turn and yield the reply text incrementally.

    Error semantics match :func:`run_chatgpt`: any failure raises
    :class:`ChatGptError`, possibly after some text has been yielded. Closing
    the generator early closes the HTTP response.
    """
    settings = settings or get_settings()
    yield from _stream_deltas(prompt, settings)


def _stream_deltas(prompt: str, settings: Settings) -> Iterator[str]:
    with _chatgpt_slot(settings):
        response = _open_stream(prompt, settings)
        parser = ChatGptStreamParser()
        try:
            for event_type, data in iter_sse(response):
                yield from parser.feed(event_type, data)
        except TimeoutError as exc:  # socket idle mid-stream
            raise ChatGptTimeoutError(
                f"chatgpt.com stream stalled past {settings.chatgpt_timeout}s."
            ) from exc
        finally:
            response.close()

        if parser.failure:
            raise ChatGptError(f"chatgpt.com reported a failure: {parser.failure}")
        if not parser.completed and not parser.emitted:
            raise ChatGptError("chatgpt.com stream ended without any reply text.")
