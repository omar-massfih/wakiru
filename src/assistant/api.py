"""FastAPI surface for the assistant."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from . import telegram
from .agent import build_agent
from .calendar import run_reminders
from .calendar.context import resolve_tz, upcoming_events
from .chat import run_chat, run_upkeep
from .codex_runner import CodexError
from .config import get_settings
from .memory import consolidate_memory, store

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


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    """
    fired = run_reminders(get_settings())
    return {"count": len(fired), "fired": fired}


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
