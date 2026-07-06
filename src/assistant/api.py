"""FastAPI surface for the assistant."""

from __future__ import annotations

import itertools
import logging
import uuid
from functools import lru_cache

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .agent import build_agent
from .codex_runner import CodexError
from .config import get_settings
from .memory import consolidate_memory, store, update_memory

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic assistant", version="0.1.0")

# Simple in-process turn counter driving periodic consolidation.
_turn_counter = itertools.count(1)


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

    # Periodic consolidation ("sleep"), also in the background.
    settings = get_settings()
    every = settings.consolidate_every_n_turns
    if every > 0 and next(_turn_counter) % every == 0:
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
