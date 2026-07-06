"""LangGraph orchestration layer — the assistant's brain wiring.

Graph shape::

    START -> recall -> agenda -> codex -> END

* **recall** — semantic memory lookup for the latest user turn; the retrieved
  context is stashed ephemerally (overwritten each turn, never accumulated) and
  the recalled notes are reinforced.
* **agenda** — the current time plus upcoming calendar events, stashed
  ephemerally so the model has a clock and knows what's scheduled.
* **codex** — feed ``[recall context] + [agenda] + history`` to the Codex model.

Working memory is bounded *off* the reply path: after the reply is sent, the API
layer runs :func:`maybe_summarize` in the background, which folds older turns
into a rolling summary and trims history via ``update_state`` — so a long thread
never pays for summarization latency inline.

The graph is compiled with a SQLite checkpointer (a *separate* DB from the vector
index), so conversation history persists per ``thread_id`` (working memory). On
build we ``reindex`` the vector store from the markdown files so hand-edits and
embedding-model changes self-heal. Long-term memory upkeep — saving, updating, and
forgetting notes, plus periodic consolidation — is kicked off in the background by
the API layer, off the reply path.
"""

from __future__ import annotations

import logging
import sqlite3

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .calendar import agenda_context
from .config import Settings, get_settings
from .llm import build_model
from .memory import index, recall_context

logger = logging.getLogger(__name__)


class BrainState(MessagesState):
    """Conversation state plus ephemeral per-turn recall + a rolling summary."""

    recall: str
    agenda: str
    summary: str


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def summarize_fold(
    settings: Settings,
    model,
    messages: list[BaseMessage],
    summary: str,
) -> dict | None:
    """Fold older turns into a rolling summary once history grows too long.

    Returns a state update (``summary`` + ``RemoveMessage`` list) or ``None``
    when history is under the threshold or the model call fails (in which case
    the full history is kept and the next pass retries).
    """
    limit = settings.working_memory_max_messages
    if limit <= 0 or len(messages) <= limit:
        return None
    keep = settings.working_memory_keep_recent
    # keep<=0 means summarize everything; messages[:-0] would wrongly be empty.
    older = messages if keep <= 0 else messages[:-keep]
    transcript = "\n".join(
        f"{m.type}: {m.content if isinstance(m.content, str) else str(m.content)}"
        for m in older
    )
    instruction = (
        "Summarize the earlier conversation below into a concise running "
        "summary that preserves durable facts, decisions, and open threads. "
        "Fold in the existing summary if present.\n\n"
        f"Existing summary: {summary or '(none)'}\n\n"
        f"Earlier conversation:\n{transcript}"
    )
    try:
        new_summary = model.invoke([HumanMessage(content=instruction)]).content
    except Exception:
        logger.exception("working-memory summarization failed; keeping history")
        return None
    if not isinstance(new_summary, str):
        new_summary = str(new_summary)
    removals: list[BaseMessage] = []
    for m in older:
        if m.id is None:
            # The add_messages reducer assigns every checkpointed message an id,
            # so this should be unreachable; a skipped message would linger.
            logger.warning("cannot fold message without an id: %.80r", m.content)
            continue
        removals.append(RemoveMessage(id=m.id))
    return {"summary": new_summary, "messages": removals}


def maybe_summarize(
    agent: CompiledStateGraph, settings: Settings, thread_id: str
) -> None:
    """Bound one thread's working memory in the background (best-effort).

    Runs after the reply has been sent. If the user's next turn lands before the
    fold does, the removals still target the older messages by their stable ids
    and the summary simply lags one turn — harmless for a single user.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = agent.get_state(config)
        update = summarize_fold(
            settings,
            build_model(settings),
            snapshot.values.get("messages", []),
            snapshot.values.get("summary", ""),
        )
        if update is not None:
            agent.update_state(config, update, as_node="codex")
    except Exception:
        logger.exception("background summarization failed for thread %s", thread_id)


def _checkpointer(settings: Settings) -> SqliteSaver:
    conn = sqlite3.connect(settings.checkpoints_db_path, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return SqliteSaver(conn)


def build_agent(settings: Settings | None = None) -> CompiledStateGraph:
    """Build and compile the assistant graph (with memory wired in)."""
    settings = settings or get_settings()
    settings.memory_path.mkdir(parents=True, exist_ok=True)

    # Self-heal the vector index from the files on disk (picks up hand-edits and
    # migrates automatically if the embedding model changed).
    try:
        index.reindex(settings)
    except Exception:
        logger.exception("startup reindex failed; continuing with existing index")

    model = build_model(settings)

    def recall(state: BrainState) -> dict:
        query = _latest_human_text(state["messages"])
        context = recall_context(settings, query)
        return {"recall": context.content}

    def agenda(state: BrainState) -> dict:
        """Give the model a clock and today's schedule (ephemeral, per turn)."""
        if not settings.enable_calendar:
            return {"agenda": ""}
        try:
            return {"agenda": agenda_context(settings)}
        except Exception:
            logger.exception("building agenda context failed; continuing without it")
            return {"agenda": ""}

    def call_codex(state: BrainState) -> dict:
        prefix: list[BaseMessage] = []
        if state.get("recall"):
            prefix.append(SystemMessage(content=state["recall"]))
        if state.get("agenda"):
            prefix.append(SystemMessage(content=state["agenda"]))
        if state.get("summary"):
            prefix.append(
                SystemMessage(content="Conversation so far:\n" + state["summary"])
            )
        reply = model.invoke(prefix + list(state["messages"]))
        return {"messages": [reply]}

    graph = StateGraph(BrainState)
    graph.add_node("recall", recall)
    graph.add_node("agenda", agenda)
    graph.add_node("codex", call_codex)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "agenda")
    graph.add_edge("agenda", "codex")
    graph.add_edge("codex", END)
    return graph.compile(checkpointer=_checkpointer(settings))
