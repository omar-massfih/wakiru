"""FastAPI surface for the assistant."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import slack, telegram, webui
from .agent import build_agent
from .briefing import run_briefing
from .calendar import remote as calendar_remote
from .calendar import run_reminders
from .calendar import sync as calendar_sync
from .calendar.context import now, resolve_tz, upcoming_events
from .chat import run_chat, run_chat_stream, run_upkeep
from .config import get_settings
from .docs import extract as docs_extract
from .docs import store as docs_store
from .docs.summarize import summarize_document
from .heartbeat import next_wake_at as heartbeat_next_wake
from .heartbeat import run_heartbeat
from .mail import client as mail_client
from .mail import snapshot as mail_snapshot
from .mail.client import MailDisabledError
from .memory import consolidate_memory, store
from .sleep import run_sleep
from .tasks import store as tasks_store
from .tasks.reminders import run_task_reminders

logger = logging.getLogger(__name__)


async def _ticker(label: str, worker: Callable[[], object], interval: Callable[[], float]) -> None:
    """Run ``worker`` forever on a wall-clock cadence, independent of chat traffic.

    The workers are synchronous (SQLite + urllib), so each tick runs in a worker
    thread to keep the event loop free. Best-effort: any error is logged and the
    loop keeps ticking; the workers' own dedupe ledgers / idempotent upserts make
    every pass safe to repeat. ``interval`` is a callable so a tick always sleeps
    on the current settings.
    """
    while True:
        try:
            await asyncio.to_thread(worker)
        except Exception:
            logger.exception("%s tick failed", label)
        await asyncio.sleep(interval())


def _reminder_tick_once() -> None:
    """One reminder pass plus the cheap jobs that ride the same tick."""
    # The agent rides along so each delivered push is also recorded
    # into the authorized chats' working memory (proactive loop-in).
    run_reminders(get_settings(), _agent())
    run_task_reminders(get_settings(), _agent())
    # The daily briefing rides the same tick; its own ledger makes it
    # exactly-once per day and a cheap no-op every other pass.
    run_briefing(get_settings(), agent=_agent())
    # The unread-mail snapshot rides along too, on its own (slower)
    # cadence — a no-op tick when email is off or the snapshot is fresh.
    mail_snapshot.maybe_refresh(get_settings())


async def _reminder_tick_loop() -> None:
    """Fire due reminders (and ride-along jobs) on the reminder cadence."""
    await _ticker(
        "reminder", _reminder_tick_once, lambda: get_settings().reminder_tick_seconds
    )


async def _heartbeat_loop() -> None:
    """Wake the deliberative layer when it is due — a self-paced scheduler.

    Not a fixed-cadence sleep: each tick asks ``heartbeat.next_wake_at`` when the
    next wake should be (the fixed ``heartbeat_minutes`` by default, pulled
    earlier by a soon-due follow-up or a model-set ``set_next_wake``), and only
    then wakes the model. Sleeps in slices of at most 60s so a follow-up or
    self-wake scheduled mid-sleep (from a chat turn) takes effect within the
    minute. The slice is a cheap SQLite read, not a model call; every wake is
    the token-cost dial. Same best-effort discipline as the other tickers.
    """
    while True:
        try:
            settings = get_settings()
            current = now(settings)
            target = await asyncio.to_thread(heartbeat_next_wake, settings, current)
            if current >= target:
                await asyncio.to_thread(run_heartbeat, settings, _agent())
                delay = 60.0
            else:
                delay = min((target - current).total_seconds(), 60.0)
        except Exception:
            logger.exception("heartbeat tick failed")
            delay = 60.0
        await asyncio.sleep(max(delay, 1.0))


async def _sleep_loop() -> None:
    """Run the nightly memory-maintenance pass on its own slow cadence.

    Its own loop, not the reminder tick: it must run even when reminders are
    off, and its consolidation LLM step can take minutes — riding the reminder
    tick would delay minute-precise reminders. The once-per-day ledger makes
    every tick outside the due window a cheap no-op, so a 5-minute cadence just
    bounds how late after ``sleep_time`` the pass lands.
    """
    await _ticker("sleep", lambda: run_sleep(get_settings(), _agent()), lambda: 300)


async def _calendar_sync_loop() -> None:
    """Mirror the configured ICS feeds on their own (slower) cadence.

    The upsert is idempotent so an overlapping manual POST /calendar/sync is
    harmless.
    """
    await _ticker(
        "calendar sync",
        lambda: calendar_sync.pull_feeds(get_settings()),
        lambda: get_settings().calendar_sync_minutes * 60,
    )


def _caldav_sync_once(settings) -> dict:
    """One CalDAV cycle: reconcile queued pushes first, then pull the collection.

    Reconcile before pull so a locally-pending edit lands remotely before the pull
    might otherwise defer to (and then re-import) a now-stale server copy.
    """
    reconciled = calendar_sync.reconcile_caldav(settings) if settings.enable_caldav_write else {}
    pulled = calendar_sync.pull_caldav(settings)
    return {"pull": pulled, "reconcile": reconciled}


async def _caldav_sync_loop() -> None:
    """Mirror the writable CalDAV collection and drain the push outbox on a cadence."""
    await _ticker(
        "caldav sync",
        lambda: _caldav_sync_once(get_settings()),
        lambda: get_settings().caldav_sync_minutes * 60,
    )


def _log_task_death(task: asyncio.Task) -> None:
    """Surface a background task that stopped on its own — it should run forever."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("background task %r died", task.get_name(), exc_info=exc)
    else:
        logger.error("background task %r exited unexpectedly", task.get_name())


def _is_loopback_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def require_token(request: Request) -> None:
    """Gate every endpoint (except /health) behind ``API_TOKEN`` when it is set.

    Unset (the default) keeps the legacy loopback-trust behavior: anyone who can
    reach the port is trusted, which is only safe on 127.0.0.1.
    """
    token = get_settings().api_token
    if not token:
        return
    header = request.headers.get("authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(credential.strip(), token):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.api_token and not _is_loopback_host(settings.host):
        if not settings.allow_unauthenticated:
            raise RuntimeError(
                f"Refusing to serve on {settings.host} without API_TOKEN: every "
                "endpoint (including /memory, which returns personal notes) would "
                "be open to that network. Set API_TOKEN, or set "
                "ALLOW_UNAUTHENTICATED=1 to accept the exposure deliberately."
            )
        logger.warning(
            "ALLOW_UNAUTHENTICATED=1: serving on %s with no API_TOKEN — every "
            "endpoint (including /memory, which returns personal notes) is open "
            "to that network.",
            settings.host,
        )
    if settings.codex_sandbox != "read-only" and (
        settings.telegram_bot_token
        or settings.slack_bot_token
        or not _is_loopback_host(settings.host)
    ):
        logger.warning(
            "CODEX_SANDBOX=%s while a remote channel is reachable (telegram/slack/"
            "non-loopback bind) — anyone who can message the assistant can make "
            "Codex write to the filesystem.",
            settings.codex_sandbox,
        )
    if settings.enable_code_execution and (
        settings.telegram_bot_token
        or settings.slack_bot_token
        or not _is_loopback_host(settings.host)
    ):
        logger.warning(
            "ENABLE_CODE_EXECUTION=1 while a remote channel is reachable "
            "(telegram/slack/non-loopback bind) — anyone who can message the "
            "assistant can run Python in this container.",
        )
    tasks: list[asyncio.Task] = []
    if settings.enable_reminders and settings.reminder_tick_seconds > 0:
        tasks.append(asyncio.create_task(_reminder_tick_loop(), name="reminder-ticker"))
        logger.info("reminder ticker started (every %ss)", settings.reminder_tick_seconds)
    if settings.enable_heartbeat and settings.heartbeat_minutes > 0:
        tasks.append(asyncio.create_task(_heartbeat_loop(), name="heartbeat"))
        logger.info("heartbeat started (every %d min)", settings.heartbeat_minutes)
    if settings.enable_sleep:
        tasks.append(asyncio.create_task(_sleep_loop(), name="sleep"))
        logger.info("nightly sleep started (due at %s)", settings.sleep_time)
    if settings.calendar_ics_urls and settings.calendar_sync_minutes > 0:
        tasks.append(asyncio.create_task(_calendar_sync_loop(), name="calendar-sync"))
        logger.info(
            "calendar sync started (%d feed(s), every %d min)",
            len(settings.calendar_ics_urls), settings.calendar_sync_minutes,
        )
    if calendar_remote.is_configured(settings) and settings.caldav_sync_minutes > 0:
        tasks.append(asyncio.create_task(_caldav_sync_loop(), name="caldav-sync"))
        logger.info(
            "remote calendar sync started (provider=%s, write=%s, every %d min)",
            settings.caldav_provider, settings.enable_caldav_write, settings.caldav_sync_minutes,
        )
    if settings.telegram_bot_token:
        tasks.append(
            asyncio.create_task(telegram.poll_loop(_agent(), settings), name="telegram-poll")
        )
    stop_socket_mode = None
    if settings.slack_app_token and settings.slack_bot_token:
        # A failed websocket connect must not take the whole server down —
        # the HTTP channels still work without Slack.
        try:
            # to_thread: connect() blocks on the websocket handshake.
            stop_socket_mode = await asyncio.to_thread(
                slack.start_socket_mode, _agent(), settings
            )
            logger.info("slack socket mode connected")
        except Exception:
            logger.exception("slack socket mode failed to start; continuing without it")
    for task in tasks:
        task.add_done_callback(_log_task_death)
    try:
        yield
    finally:
        if stop_socket_mode is not None:
            stop_socket_mode()
        for task in tasks:
            task.cancel()
        # Wait for cancellation to land so shutdown doesn't strand mid-operation
        # work; return_exceptions swallows the resulting CancelledErrors.
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Agentic assistant", version="0.1.0", lifespan=lifespan)


@lru_cache
def _agent():
    """Build the graph once and reuse it across requests."""
    return build_agent()


# max_length caps keep a single request from queuing unbounded embedder/LLM work.
class ChatRequest(BaseModel):
    message: str = Field(max_length=100_000)
    # Continue an existing conversation by passing the id returned earlier.
    thread_id: str | None = Field(default=None, max_length=200)


class DocRequest(BaseModel):
    title: str = Field("", max_length=500)
    # Ingesting whole documents is the point — roomy enough for a book.
    text: str | None = Field(None, max_length=2_000_000)
    # Alternative to text: a page to fetch server-side (requires
    # ENABLE_DOCS_URL_INGEST). Exactly one of text/url must be given.
    url: str | None = Field(None, max_length=2_000)


class DraftRequest(BaseModel):
    to: str = Field(max_length=1_000)
    subject: str = Field(max_length=1_000)
    body: str = Field(max_length=100_000)
    # Opt in per-request to actually send. Even then the server-side
    # `enable_email_send` switch must also be on.
    send: bool = False


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    """A minimal self-contained chat page that streams from /chat/stream.

    Not token-gated: the page carries no data. The API calls it makes are — when
    API_TOKEN is set the page prompts for it and sends it as a bearer header.
    """
    return webui.PAGE


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_token)])
def chat(req: ChatRequest, background: BackgroundTasks) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        reply = run_chat(_agent(), req.message, thread_id, settings=get_settings())
    except Exception as exc:
        # Any provider's failure — codex, chatgpt.com, openai/Azure, anthropic —
        # should surface as a clean 502, not just CodexError (which would leak
        # every other provider's error out as a bare 500).
        logger.exception("chat turn failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # All post-reply maintenance — long-term memory, working-memory folding,
    # calendar extraction, periodic consolidation — runs off the request path
    # (shared with the Telegram channel) so it never adds latency.
    background.add_task(run_upkeep, _agent(), get_settings(), req.message, reply, thread_id)

    return ChatResponse(reply=reply, thread_id=thread_id)


def sse_frame(data: str, event: str | None = None) -> str:
    """Encode one Server-Sent Events frame.

    Per the SSE spec a frame's payload is carried by one ``data:`` line *per line
    of content*, and the receiver rejoins them with newlines. Emitting a raw
    ``f"data: {text}\\n\\n"`` is wrong the moment ``text`` contains a blank line:
    it terminates the frame early, and the remainder is parsed as a new
    (unrecognized) frame and dropped. Not a rare edge — the Codex provider yields
    the whole reply as one chunk, so nearly every multi-paragraph answer would
    lose everything after its first blank line.
    """
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}{lines}\n"


@app.post("/chat/stream", dependencies=[Depends(require_token)])
async def chat_stream(req: ChatRequest, background: BackgroundTasks) -> StreamingResponse:
    """Stream a reply as Server-Sent Events, then run upkeep once, off-path.

    Emits ``data:`` frames as the model produces the reply (incrementally for
    every provider, including Codex via its ``--json`` event stream), a final
    ``event: done`` frame carrying the ``thread_id``, and ``event: error`` if the
    model fails mid-stream. Post-reply maintenance runs after the stream closes,
    exactly as the buffered ``/chat`` endpoint does.
    """
    thread_id = req.thread_id or str(uuid.uuid4())

    async def event_stream():
        parts: list[str] = []
        try:
            async for chunk in run_chat_stream(
                _agent(), req.message, thread_id, settings=get_settings()
            ):
                parts.append(chunk)
                yield sse_frame(chunk)
        except Exception as exc:
            # Every provider, not just codex, must surface as an error frame
            # rather than breaking the stream with an unhandled exception (no
            # done/error frame at all) partway through.
            logger.exception("streaming chat turn failed")
            yield sse_frame(str(exc), event="error")
            return
        yield sse_frame(thread_id, event="done")
        # Upkeep needs the full reply; run it off the stream so it never delays
        # the client, mirroring the buffered /chat path.
        reply = "".join(parts)
        background.add_task(
            run_upkeep, _agent(), get_settings(), req.message, reply, thread_id
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/slack/events")
async def slack_events(request: Request, background: BackgroundTasks) -> dict:
    """Slack Events API callback.

    Not behind ``require_token``: Slack can't send a bearer token. It is
    authenticated instead by the HMAC signature over the raw body, verified
    against ``slack_signing_secret`` before anything is parsed or dispatched.
    The turn runs as a background task because Slack expects an ack within 3s.
    """
    settings = get_settings()
    if not (settings.slack_bot_token and settings.slack_signing_secret):
        raise HTTPException(status_code=404, detail="Slack channel is not configured.")

    raw = await request.body()
    if not slack.verify_signature(
        settings.slack_signing_secret,
        request.headers.get("x-slack-request-timestamp", ""),
        raw,
        request.headers.get("x-slack-signature", ""),
    ):
        raise HTTPException(status_code=401, detail="Bad Slack signature.")

    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError as exc:
        # Signature-valid but unparseable: a 500 here would make Slack retry it.
        raise HTTPException(status_code=400, detail="Malformed JSON body.") from exc
    # The one-time endpoint verification handshake.
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge", "")}

    # A retry means Slack didn't see our ack, not that the turn didn't run — the
    # first delivery may still be working in the background. Ack and drop it.
    # Read only after the signature check: an unverified header proves nothing.
    if request.headers.get("x-slack-retry-num"):
        logger.info(
            "acking slack retry %s of event %s without re-running it",
            request.headers.get("x-slack-retry-num"),
            payload.get("event_id"),
        )
        return {"ok": True}

    def turn() -> None:
        upkeep = slack.handle_event(_agent(), settings, payload)
        if upkeep is not None:
            upkeep()

    background.add_task(turn)
    return {"ok": True}


@app.get("/memory", dependencies=[Depends(require_token)])
def memory_stats() -> dict:
    """Introspect the brain: counts by kind and the current note listing."""
    settings = get_settings()
    notes = store.list_notes(settings)
    by_kind: dict[str, int] = {}
    for note in notes:
        by_kind[note.kind] = by_kind.get(note.kind, 0) + 1
    return {
        "total": len(notes),
        "by_kind": by_kind,
        "notes": [
            {
                "name": n.name,
                "kind": n.kind,
                "description": n.description,
                "salience": n.salience,
                "recall_count": n.recall_count,
                "updated": n.updated,
            }
            for n in notes
        ],
    }


@app.post("/memory/consolidate", dependencies=[Depends(require_token)])
def memory_consolidate() -> dict:
    """Trigger a consolidation pass on demand and return what changed."""
    return consolidate_memory(get_settings())


@app.post("/sleep/run", dependencies=[Depends(require_token)])
def sleep_run() -> dict:
    """Run the nightly memory-maintenance pass now (bypasses the time-of-day gate).

    Idempotent per local date via the fired ledger: a second call the same day
    reports it already ran rather than re-consolidating. The LLM step is still
    skipped when no episode is newer than the last pass.
    """
    return run_sleep(get_settings(), agent=_agent(), force=True)


@app.post("/reminders/run", dependencies=[Depends(require_token)])
def reminders_run() -> dict:
    """Fire any reminders now due and return what was sent.

    The in-process ticker calls this same logic on a cadence; this endpoint lets it
    also be driven manually or from external cron. Idempotent via the dedupe ledger.
    Fires both calendar-event and due-task reminders.
    """
    settings = get_settings()
    fired = run_reminders(settings, _agent()) + run_task_reminders(settings, _agent())
    return {"count": len(fired), "fired": fired}


@app.post("/calendar/sync", dependencies=[Depends(require_token)])
def calendar_sync_run() -> dict:
    """Pull ICS feeds now, and (when enabled) run the CalDAV pull + outbox reconcile.

    The in-process tickers call the same logic on their cadences; this endpoint lets
    both be driven manually or from external cron. All steps are idempotent.
    """
    settings = get_settings()
    result: dict = {"feeds": calendar_sync.pull_feeds(settings)}
    if calendar_remote.is_configured(settings):
        result["caldav"] = _caldav_sync_once(settings)
    return result


@app.post("/briefing/run", dependencies=[Depends(require_token)])
def briefing_run() -> dict:
    """Send today's briefing now (even if the scheduled time hasn't passed).

    Idempotent per local date via the fired ledger: a second call the same day
    reports it was already sent rather than pushing a duplicate.
    """
    return run_briefing(get_settings(), force=True, agent=_agent())


@app.post("/heartbeat/run", dependencies=[Depends(require_token)])
def heartbeat_run() -> dict:
    """Run one heartbeat now (bypasses the ambient throttle, not quiet hours).

    The deliberative wake: triage the situation, let the model decide whether
    to reach out, and report what happened. Due followups are consumed
    exactly-once, so a manual run replaces — not duplicates — the scheduled one.
    """
    return run_heartbeat(get_settings(), agent=_agent(), force=True)


@app.post("/documents", dependencies=[Depends(require_token)])
def docs_add(req: DocRequest) -> dict:
    """Ingest a document (chunked + embedded) so it can be recalled and summarized.

    Give either ``text`` (with a ``title``) or — when ``ENABLE_DOCS_URL_INGEST``
    is on — a ``url`` fetched server-side (HTML reduced to prose; the page's
    ``<title>`` becomes the title unless one is given).
    """
    settings = get_settings()
    if (req.text is None) == (req.url is None):
        raise HTTPException(status_code=422, detail="Give exactly one of 'text' or 'url'.")
    title = req.title
    if req.url is not None:
        if not settings.enable_docs_url_ingest:
            raise HTTPException(
                status_code=403,
                detail="URL ingestion is off. Set ENABLE_DOCS_URL_INGEST=true to allow it.",
            )
        try:
            fetched_title, text = docs_extract.fetch_url_text(req.url)
        except docs_extract.ExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        title = title or fetched_title
    else:
        assert req.text is not None  # the exactly-one gate above
        text = req.text
    doc = docs_store.add_document(settings, title, text)
    return {"id": doc.id, "title": doc.title, "chunks": doc.chunks, "added": doc.added}


@app.post("/documents/upload", dependencies=[Depends(require_token)])
async def docs_upload(file: UploadFile, title: str = Form("")) -> dict:
    """Ingest an uploaded file (PDF, DOCX, or any text-like format)."""
    settings = get_settings()
    # Read in bounded chunks: multipart bodies bypass the pydantic max_length
    # caps, so an unmetered read() would buffer an arbitrarily large upload.
    limit = settings.docs_upload_max_bytes
    pieces: list[bytes] = []
    received = 0
    while chunk := await file.read(1 << 20):
        received += len(chunk)
        if received > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {limit}-byte limit (DOCS_UPLOAD_MAX_BYTES).",
            )
        pieces.append(chunk)
    content = b"".join(pieces)
    try:
        text = docs_extract.extract_text(file.filename or "", content)
    except docs_extract.ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    doc = docs_store.add_document(settings, title or file.filename or "", text)
    return {"id": doc.id, "title": doc.title, "chunks": doc.chunks, "added": doc.added}


@app.get("/documents", dependencies=[Depends(require_token)])
def docs_list() -> dict:
    """List ingested documents (metadata only, most recent first)."""
    items = docs_store.list_documents(get_settings())
    return {
        "total": len(items),
        "documents": [
            {"id": d.id, "title": d.title, "chunks": d.chunks, "added": d.added}
            for d in items
        ],
    }


@app.get("/documents/search", dependencies=[Depends(require_token)])
def docs_search(q: str = Query(max_length=1_000)) -> dict:
    """Return the document chunks most relevant to ``q``."""
    chunks = docs_store.search_chunks(get_settings(), q)
    return {
        "total": len(chunks),
        "chunks": [
            {"doc_id": c.doc_id, "doc_title": c.doc_title, "text": c.text, "similarity": c.similarity}
            for c in chunks
        ],
    }


@app.post("/documents/{doc_id}/summarize", dependencies=[Depends(require_token)])
def docs_summarize(doc_id: str) -> dict:
    """Summarize a stored document with the configured model."""
    try:
        summary = summarize_document(get_settings(), doc_id)
    except Exception as exc:
        # Provider-agnostic, like /chat: any model failure is a 502, not a 500.
        logger.exception("document summarization failed for %s", doc_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if summary is None:
        raise HTTPException(status_code=404, detail="No such document.")
    return {"id": doc_id, "summary": summary}


@app.delete("/documents/{doc_id}", dependencies=[Depends(require_token)])
def docs_delete(doc_id: str) -> dict:
    """Delete a document and its chunks."""
    if not docs_store.delete_document(get_settings(), doc_id):
        raise HTTPException(status_code=404, detail="No such document.")
    return {"id": doc_id, "deleted": True}


@app.get("/email", dependencies=[Depends(require_token)])
def email_list(unread_only: bool = True) -> dict:
    """List recent INBOX messages (headers only; never marks them read)."""
    try:
        messages = mail_client.list_recent(get_settings(), unread_only=unread_only)
    except MailDisabledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "total": len(messages),
        "messages": [
            {"uid": m.uid, "sender": m.sender, "subject": m.subject, "date": m.date}
            for m in messages
        ],
    }


@app.get("/email/{uid}", dependencies=[Depends(require_token)])
def email_read(uid: str) -> dict:
    """Read one message with its plain-text body (leaves it unread)."""
    try:
        message = mail_client.read_message(get_settings(), uid)
    except MailDisabledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if message is None:
        raise HTTPException(status_code=404, detail="No such message.")
    return {
        "uid": message.uid,
        "sender": message.sender,
        "subject": message.subject,
        "date": message.date,
        "body": message.body,
        "attachments": message.attachments,
    }


@app.post("/email/draft", dependencies=[Depends(require_token)])
def email_draft(req: DraftRequest) -> dict:
    """Save a draft — or send it, if the caller AND the server both opt in.

    Drafting is the default. Sending requires ``send=true`` here *and*
    ``ENABLE_EMAIL_SEND=true`` on the server; otherwise this returns 409 and
    nothing leaves the mailbox.
    """
    settings = get_settings()
    try:
        if req.send:
            result = mail_client.send_message(settings, req.to, req.subject, req.body)
            return {"sent": True, "summary": result}
        result = mail_client.save_draft(settings, req.to, req.subject, req.body)
        return {"sent": False, "summary": result}
    except MailDisabledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tasks", dependencies=[Depends(require_token)])
def tasks(include_done: bool = False) -> dict:
    """List tasks — open ones by default, or all with ``?include_done=true``."""
    settings = get_settings()
    items = tasks_store.list_tasks(settings, include_done=include_done)
    return {
        "total": len(items),
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "done": t.done,
                "due": t.due,
                "notes": t.notes,
                "done_at": t.done_at,
            }
            for t in items
        ],
    }


@app.get("/goals", dependencies=[Depends(require_token)])
def goals_list() -> dict:
    """List the assistant's open standing goals (see assistant.goals)."""
    from . import goals as goals_store

    settings = get_settings()
    items = goals_store.list_open(settings)
    return {
        "total": len(items),
        "goals": [
            {
                "id": g.id,
                "title": g.title,
                "state": g.state,
                "next_action_at": g.next_action_at,
                "created_at": g.created_at,
                "updated_at": g.updated_at,
            }
            for g in items
        ],
    }


@app.get("/calendar", dependencies=[Depends(require_token)])
def calendar() -> dict:
    """List upcoming events (within the configured horizon) and the current time."""
    settings = get_settings()
    events = upcoming_events(settings)
    return {
        "now": datetime.now(resolve_tz(settings)).isoformat(timespec="seconds"),
        "total": len(events),
        "events": [
            {
                "id": e.id,
                "title": e.title,
                "start": e.start,
                "end": e.end,
                "location": e.location,
                "notes": e.notes,
            }
            for e in events
        ],
    }
