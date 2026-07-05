"""LangGraph orchestration layer.

A minimal single-node graph today: it feeds the running message history to the
Codex-backed model and appends the reply. This is the extension point — add nodes,
routing, or a checkpointer here as the assistant grows.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .llm import build_model


def build_agent() -> CompiledStateGraph:
    """Build and compile the assistant graph."""
    model = build_model()

    def call_codex(state: MessagesState) -> dict[str, list[BaseMessage]]:
        reply = model.invoke(state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("codex", call_codex)
    graph.add_edge(START, "codex")
    graph.add_edge("codex", END)
    return graph.compile()
