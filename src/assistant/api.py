"""FastAPI surface for the assistant."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .agent import build_agent, maybe_summarize
from .calendar import run_reminders, update_calendar
from .calendar.context import resolve_tz, upcoming_events
from .codex_runner import CodexError
from .config import get_settings
from .memory import consolidate_memory, index, store, update_memory

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    task = None
    if settings.enable_reminders and settings.reminder_tick_seconds > 0:
        task = asyncio.create_task(_reminder_tick_loop())
        logger.info("reminder ticker started (every %ss)", settings.reminder_tick_seconds)
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background: BackgroundTasks) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = _agent().invoke(
            {"messages": [HumanMessage(content=req.message)]}, config=config
        )
    except CodexError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    reply = result["messages"][-1].content
    if not isinstance(reply, str):
        reply = str(reply)

    # Long-term memory upkeep off the request path so it never adds latency: an
    # episodic trace plus a reconciling LLM extraction (save/update/forget).
    background.add_task(update_memory, None, req.message, reply, thread_id)

    # Working-memory upkeep, also off the request path: fold older turns into
    # the rolling summary once this thread's history grows past the threshold.
    settings = get_settings()
    background.add_task(maybe_summarize, _agent(), settings, thread_id)

    # Calendar upkeep, also off the request path: a reconciling LLM extraction
    # that creates/reschedules/cancels events from the turn.
    if settings.enable_calendar and settings.enable_auto_schedule:
        background.add_task(update_calendar, None, req.message, reply)

    # Periodic consolidation ("sleep"), also in the background. The counter is
    # persisted in the index DB so the cadence survives server restarts.
    every = settings.consolidate_every_n_turns
    if every > 0 and index.bump_turn_counter(settings) % every == 0:
        background.add_task(consolidate_memory, None)

    return ChatResponse(reply=reply, thread_id=thread_id)


@app.get("/memory")
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


@app.post("/memory/consolidate")
def memory_consolidate() -> dict:
    """Trigger a consolidation pass on demand and return what changed."""
    return consolidate_memory(get_settings())


@app.post("/reminders/run")
def reminders_run() -> dict:
    """Fire any reminders now due and return what was sent.

    The in-process ticker calls this same logic on a cadence; this endpoint lets it
    also be driven manually or from external cron. Idempotent via the dedupe ledger.
    """
    fired = run_reminders(get_settings())
    return {"count": len(fired), "fired": fired}


@app.get("/calendar")
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
