"""FastAPI surface for the assistant."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import slack, telegram, webui
from .agent import build_agent
from .calendar import run_reminders
from .calendar.context import resolve_tz, upcoming_events
from .chat import run_chat, run_chat_stream, run_upkeep
from .codex_runner import CodexError
from .config import get_settings
from .docs import store as docs_store
from .docs.summarize import summarize_document
from .mail import client as mail_client
from .mail.client import MailDisabledError
from .memory import consolidate_memory, store
from .tasks import store as tasks_store
from .tasks.reminders import run_task_reminders

logger = logging.getLogger(__name__)


async def _reminder_tick_loop() -> None:
    """Fire due reminders on a wall-clock cadence, independent of chat traffic.

    ``run_reminders`` is synchronous (SQLite + a urllib POST), so it runs in a
    worker thread to keep the event loop free. Best-effort: any error is logged and
    the loop keeps ticking. The dedupe ledger makes each pass idempotent.
    """
    while True:
        try:
            await asyncio.to_thread(run_reminders, get_settings())
            await asyncio.to_thread(run_task_reminders, get_settings())
        except Exception:
            logger.exception("reminder tick failed")
        await asyncio.sleep(get_settings().reminder_tick_seconds)


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
        logger.warning(
            "API_TOKEN is not set while binding to %s — every endpoint (including "
            "/memory, which returns personal notes) is open to that network. "
            "Set API_TOKEN before exposing the server.",
            settings.host,
        )
    tasks: list[asyncio.Task] = []
    if settings.enable_reminders and settings.reminder_tick_seconds > 0:
        tasks.append(asyncio.create_task(_reminder_tick_loop(), name="reminder-ticker"))
        logger.info("reminder ticker started (every %ss)", settings.reminder_tick_seconds)
    if settings.telegram_bot_token:
        tasks.append(
            asyncio.create_task(telegram.poll_loop(_agent(), settings), name="telegram-poll")
        )
    for task in tasks:
        task.add_done_callback(_log_task_death)
    try:
        yield
    finally:
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


class ChatRequest(BaseModel):
    message: str
    # Continue an existing conversation by passing the id returned earlier.
    thread_id: str | None = None


class DocRequest(BaseModel):
    title: str
    text: str


class DraftRequest(BaseModel):
    to: str
    subject: str
    body: str
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
    except CodexError as exc:
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

    Emits ``data:`` frames as the model produces the reply (a single frame for
    the Codex provider, which can't stream token-by-token), a final
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
        except CodexError as exc:
            logger.error("streaming chat turn failed: %s", exc)
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


@app.post("/reminders/run", dependencies=[Depends(require_token)])
def reminders_run() -> dict:
    """Fire any reminders now due and return what was sent.

    The in-process ticker calls this same logic on a cadence; this endpoint lets it
    also be driven manually or from external cron. Idempotent via the dedupe ledger.
    Fires both calendar-event and due-task reminders.
    """
    settings = get_settings()
    fired = run_reminders(settings) + run_task_reminders(settings)
    return {"count": len(fired), "fired": fired}


@app.post("/documents", dependencies=[Depends(require_token)])
def docs_add(req: DocRequest) -> dict:
    """Ingest a document (chunked + embedded) so it can be recalled and summarized."""
    doc = docs_store.add_document(get_settings(), req.title, req.text)
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
def docs_search(q: str) -> dict:
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
    except CodexError as exc:
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
