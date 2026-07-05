"""FastAPI surface for the assistant."""

from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .agent import build_agent
from .codex_runner import CodexError
from .memory import update_memory

app = FastAPI(title="Agentic assistant", version="0.1.0")


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

    # Long-term memory upkeep (save/forget) off the request path so it never adds
    # latency. An LLM extraction handles both explicit "remember/forget …" and
    # proactively captured facts. No-ops when ENABLE_AUTO_MEMORY is false.
    background.add_task(update_memory, None, req.message, reply)

    return ChatResponse(reply=reply, thread_id=thread_id)
