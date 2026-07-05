"""LangGraph orchestration layer — the assistant's brain wiring.

Graph shape::

    START -> recall -> codex -> END

* **recall** — semantic memory lookup for the latest user turn; the retrieved
  context is stashed ephemerally (overwritten each turn, never accumulated).
* **codex** — feed ``[recall context] + history`` to the Codex-backed model.

The graph is compiled with a SQLite checkpointer, so conversation history
persists per ``thread_id`` (working memory). Long-term memory upkeep — saving and
forgetting notes, driven by an LLM extraction over the turn — is kicked off in
the background by the API layer, off the reply path.
"""

from __future__ import annotations

import sqlite3

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import Settings, get_settings
from .llm import build_model
from .memory import recall_context


class BrainState(MessagesState):
    """Conversation state plus an ephemeral, per-turn recalled-memory block."""

    recall: str


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _checkpointer(settings: Settings) -> SqliteSaver:
    conn = sqlite3.connect(settings.memory_db_path, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 5000")
    return SqliteSaver(conn)


def build_agent(settings: Settings | None = None) -> CompiledStateGraph:
    """Build and compile the assistant graph (with memory wired in)."""
    settings = settings or get_settings()
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    model = build_model(settings)

    def recall(state: BrainState) -> dict:
        query = _latest_human_text(state["messages"])
        context = recall_context(settings, query)
        return {"recall": context.content}

    def call_codex(state: BrainState) -> dict:
        prefix = [SystemMessage(content=state["recall"])] if state.get("recall") else []
        reply = model.invoke(prefix + list(state["messages"]))
        return {"messages": [reply]}

    graph = StateGraph(BrainState)
    graph.add_node("recall", recall)
    graph.add_node("codex", call_codex)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "codex")
    graph.add_edge("codex", END)
    return graph.compile(checkpointer=_checkpointer(settings))
