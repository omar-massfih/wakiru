"""FastAPI surface for the assistant."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .agent import build_agent
from .codex_runner import CodexError

app = FastAPI(title="Agentic assistant", version="0.1.0")


@lru_cache
def _agent():
    """Build the graph once and reuse it across requests."""
    return build_agent()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = _agent().invoke({"messages": [HumanMessage(content=req.message)]})
    except CodexError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    reply = result["messages"][-1].content
    if not isinstance(reply, str):
        reply = str(reply)
    return ChatResponse(reply=reply)
