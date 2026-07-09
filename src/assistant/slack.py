"""Slack channel — the Events API bridge, over the shared chat core.

Stdlib-only (``urllib`` + ``hmac``), like :mod:`assistant.telegram` and
:mod:`assistant.notify`. Slack pushes events to ``POST /slack/events`` (wired in
:mod:`assistant.api`), so unlike Telegram's long polling this one needs a public
HTTPS URL.

Security, in layers:

* **Signature.** Every callback carries an HMAC of its raw body, keyed by the
  app's signing secret. :func:`verify_signature` checks it in constant time and
  rejects stale timestamps, so a replayed or forged request never reaches the model.
* **Allowlist.** Only user ids in ``slack_allowed_user_ids`` are answered. Empty
  means *nobody* — there is no pairing handshake here, so an unconfigured
  allowlist fails closed rather than answering the whole workspace.
* **Bot loop guard.** Messages from bots (including our own) are ignored, so a
  reply can never trigger another reply.
* **Delivery dedupe.** Slack redelivers a callback it thinks we missed. Each
  envelope's ``event_id`` is claimed once (see :func:`already_seen`), so a
  redelivery can't run the turn — and its memory and calendar writes — twice.

Each user maps to a stable thread (``slack:<channel>:<user>``), so the
conversation — working memory and rolling summary — survives restarts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable

from langgraph.graph.state import CompiledStateGraph

from .chat import run_chat, run_upkeep
from .codex_runner import CodexError
from .config import Settings

logger = logging.getLogger(__name__)

_API_URL = "https://slack.com/api/chat.postMessage"
_TIMEOUT_SECONDS = 10
# Reject callbacks older than this; Slack's own guidance for replay protection.
_MAX_SKEW_SECONDS = 60 * 5

# Envelope ids already handled, newest last. Bounded so it can't grow without
# limit; Slack gives up retrying long before this many events pass through.
# Guarded by a lock: handle_event runs on FastAPI's threadpool, not the loop.
_SEEN_MAX = 1024
_seen_events: OrderedDict[str, None] = OrderedDict()
_seen_lock = threading.Lock()


def already_seen(event_id: str) -> bool:
    """Claim ``event_id``; True when this callback was already handled.

    Process-local by design: Slack's retry window is minutes, so a restart losing
    the set costs at most one duplicate reply, and this keeps the hot path free of
    a database round-trip. An empty id (a payload shape without one) never dedupes.
    """
    if not event_id:
        return False
    with _seen_lock:
        if event_id in _seen_events:
            return True
        _seen_events[event_id] = None
        if len(_seen_events) > _SEEN_MAX:
            _seen_events.popitem(last=False)
    return False


def verify_signature(
    signing_secret: str, timestamp: str, raw_body: bytes, signature: str
) -> bool:
    """Whether a Slack callback's ``X-Slack-Signature`` is authentic and fresh."""
    if not (signing_secret and timestamp and signature):
        return False
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        return False
    if age > _MAX_SKEW_SECONDS:
        return False  # replayed
    basestring = b"v0:" + timestamp.encode() + b":" + raw_body
    expected = "v0=" + hmac.new(
        signing_secret.encode(), basestring, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def post_message(token: str, channel: str, text: str) -> None:
    """Post a message to a Slack channel (best-effort; raises on transport error)."""
    body = json.dumps({"channel": channel, "text": text}).encode("utf-8")
    request = urllib.request.Request(
        _API_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read() or b"{}")
    if not payload.get("ok"):
        logger.warning("slack chat.postMessage failed: %s", payload.get("error"))


def authorized_users(settings: Settings) -> list[str]:
    """The Slack user ids this bot will answer. Empty means nobody."""
    return list(settings.slack_allowed_user_ids)


def _is_user_message(event: dict) -> bool:
    """Whether an event is a real human message (not a bot, edit, or join notice)."""
    if event.get("type") != "message":
        return False
    if event.get("bot_id") or event.get("subtype"):
        return False  # our own replies, edits, joins — never answer these
    return bool(event.get("user") and event.get("text"))


def handle_event(
    agent: CompiledStateGraph, settings: Settings, payload: dict
) -> Callable[[], None] | None:
    """Answer one Slack event callback; return its post-reply upkeep, or ``None``.

    The caller (the API route) runs the returned callable off the request path —
    Slack expects an ack within 3 seconds, and a turn takes far longer.
    """
    token = settings.slack_bot_token
    event = payload.get("event") or {}
    if token is None or not _is_user_message(event):
        return None

    user, channel, text = event["user"], event.get("channel", ""), event["text"]
    allowed = authorized_users(settings)
    if user not in allowed:
        # Fail closed: an empty allowlist answers no one.
        logger.warning("ignoring slack message from unauthorized user %s", user)
        return None

    # Claim the envelope last, once we know we'd act on it: a redelivered
    # callback must not answer twice, nor re-run the turn's memory/calendar upkeep.
    if already_seen(str(payload.get("event_id", ""))):
        logger.info("ignoring duplicate slack event %s", payload.get("event_id"))
        return None

    thread_id = f"slack:{channel}:{user}"
    try:
        reply = run_chat(agent, text, thread_id, settings=settings)
    except CodexError as exc:
        logger.error("slack chat turn failed: %s", exc)
        post_message(token, channel, "Sorry — I hit an error answering that. Try again.")
        return None
    post_message(token, channel, reply)
    return lambda: run_upkeep(agent, settings, text, reply, thread_id)
