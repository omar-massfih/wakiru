"""LangGraph orchestration layer — the assistant's brain wiring.

Graph shape::

    START -> recall -> agenda -> tasks -> profile -> agent
    agent  -> tools -> agent   (while the model calls tools, up to a cap)
    agent  -> END              (a plain-text reply ends the turn)

* **recall** — semantic memory lookup for the current turn (the latest message
  expanded with recent context, so follow-ups retrieve well); the retrieved
  context is stashed ephemerally (overwritten each turn, never accumulated) and
  the recalled notes are reinforced.
* **agenda** — the current time plus upcoming calendar events, stashed
  ephemerally so the model has a clock and knows what's scheduled.
* **tasks** — the open to-do list, stashed ephemerally so the model knows what's
  outstanding.
* **profile** — the user's durable preferences (notes tagged ``profile``:
  working hours, locations, quiet hours, tone), so replies and scheduling
  suggestions fit the person.
* **agent** — feed ``[recall] + [profile] + [agenda] + [tasks] + history`` to
  the model with the tool registry bound (:mod:`assistant.tools`).
* **tools** — execute the model's tool calls (calendar, tasks, memory, docs,
  email) through the same guarded write paths the old background extractors
  used, so the undo ledger and ambiguity guards keep working. The loop is
  bounded by ``tool_max_rounds``; past it, pending calls are answered with a
  budget-exhausted result and the next model pass runs tool-less, so history
  never ends on a dangling tool call.

``ENABLE_TOOL_LOOP=false`` restores the previous behavior: no tools are bound,
the graph is effectively the old linear shape, and the background calendar/task
extractors in :mod:`assistant.chat` take over again.

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

import atexit
import logging
import sqlite3
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .calendar import agenda_context
from .config import Settings, get_settings
from .docs import docs_context
from .docs import store as docs_store
from .llm import build_model
from .memory import index, recall_context
from .memory.profile import profile_context
from .tasks import tasks_context
from .tools import ToolContext, available_tools, execute_tool

logger = logging.getLogger(__name__)


class BrainState(MessagesState):
    """Conversation state plus ephemeral per-turn recall + a rolling summary."""

    recall: str
    agenda: str
    tasks: str
    profile: str
    summary: str
    batch_id: str
    tool_rounds: int
    tools_exhausted: bool


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def expanded_recall_query(
    messages: list[BaseMessage], summary: str, settings: Settings
) -> str:
    """The embedding query for recall: the latest message plus recent context.

    A pronoun-only follow-up ("what about the second one?") embeds poorly on
    its own; folding in snippets of the last few turns and the tail of the
    rolling summary carries the referent into the query. Bounded by
    ``recall_context_extra_chars`` so the query stays a query, not a transcript.
    """
    latest = _latest_human_text(messages)
    remaining = settings.recall_context_extra_chars
    if remaining <= 0:
        return latest

    supplement: list[str] = []
    seen_latest = False
    for message in reversed(messages):
        if len(supplement) >= max(settings.recall_context_messages, 0):
            break
        if isinstance(message, HumanMessage) and not seen_latest:
            seen_latest = True  # the latest human turn already leads the query
            continue
        if not isinstance(message, (HumanMessage, AIMessage)):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        snippet = content.strip()[:160]
        if not snippet or len(snippet) > remaining:
            continue
        supplement.append(snippet)
        remaining -= len(snippet)
    supplement.reverse()

    if summary and remaining > 0:
        tail = summary.strip()[-min(300, remaining):]
        if tail:
            supplement.append(tail)

    if not supplement:
        return latest
    return "\n".join([latest, *supplement]) if latest else "\n".join(supplement)


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
    if keep <= 0:
        older = messages
    else:
        split = len(messages) - keep
        # Never let the kept history open on a ToolMessage: its calling
        # AIMessage would be folded away, leaving an orphaned tool result
        # (which the native providers reject outright on the next turn).
        while split < len(messages) and isinstance(messages[split], ToolMessage):
            split += 1
        older = messages[:split]
    if not older:
        return None
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
            agent.update_state(config, update, as_node="agent")
    except Exception:
        logger.exception("background summarization failed for thread %s", thread_id)


def _checkpointer(settings: Settings):
    if settings.storage_backend == "postgres":
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is required when STORAGE_BACKEND=postgres")
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - depends on deployment extras
            raise RuntimeError(
                "STORAGE_BACKEND=postgres requires psycopg[binary] and "
                "langgraph-checkpoint-postgres"
            ) from exc
        # A pool, not a single connection: serverless Postgres (Neon) drops idle
        # connections, killing a lone long-lived one between turns. check=
        # revalidates on checkout and replaces dead connections transparently.
        pool = ConnectionPool(
            settings.database_url,
            min_size=0,
            max_size=4,
            open=True,
            check=ConnectionPool.check_connection,
            # Match PostgresSaver.from_conn_string's connection settings.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        atexit.register(pool.close)  # its worker threads outlive shutdown otherwise
        saver = PostgresSaver(pool)
        saver.setup()
        return saver

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

    # Same for the document chunk index, rebuilt from the stored document text.
    if settings.enable_docs:
        try:
            docs_store.reindex(settings)
        except Exception:
            logger.exception("startup docs reindex failed; continuing with existing index")

    model = build_model(settings)
    specs = available_tools(settings) if settings.enable_tool_loop else []
    by_name = {spec.name: spec for spec in specs}
    bound_model = (
        model.bind_tools([spec.to_openai_tool() for spec in specs]) if specs else model
    )

    def recall(state: BrainState) -> dict:
        query = expanded_recall_query(
            state["messages"], state.get("summary", ""), settings
        )
        content = recall_context(settings, query).content
        # Fold in the most relevant document excerpts on the same channel, so
        # "what did I write about X" is answered from ingested docs too.
        try:
            docs = docs_context(settings, query)
        except Exception:
            logger.exception("document recall failed; continuing without it")
            docs = ""
        if docs:
            content = f"{content}\n\n{docs}" if content else docs
        # First node of the turn: reset the per-turn tool-loop state too.
        return {
            "recall": content,
            "batch_id": uuid.uuid4().hex,
            "tool_rounds": 0,
            "tools_exhausted": False,
        }

    def agenda(state: BrainState) -> dict:
        """Give the model a clock and today's schedule (ephemeral, per turn)."""
        if not settings.enable_calendar:
            return {"agenda": ""}
        try:
            return {"agenda": agenda_context(settings)}
        except Exception:
            logger.exception("building agenda context failed; continuing without it")
            return {"agenda": ""}

    def tasks(state: BrainState) -> dict:
        """Give the model the open to-do list (ephemeral, per turn)."""
        if not settings.enable_tasks:
            return {"tasks": ""}
        try:
            return {"tasks": tasks_context(settings)}
        except Exception:
            logger.exception("building tasks context failed; continuing without it")
            return {"tasks": ""}

    def profile(state: BrainState) -> dict:
        """Give the model the user's durable preferences (ephemeral, per turn)."""
        try:
            return {"profile": profile_context(settings)}
        except Exception:
            logger.exception("building profile context failed; continuing without it")
            return {"profile": ""}

    def call_agent(state: BrainState) -> dict:
        prefix: list[BaseMessage] = []
        if state.get("recall"):
            prefix.append(SystemMessage(content=state["recall"]))
        if state.get("profile"):
            prefix.append(SystemMessage(content=state["profile"]))
        if state.get("agenda"):
            prefix.append(SystemMessage(content=state["agenda"]))
        if state.get("tasks"):
            prefix.append(SystemMessage(content=state["tasks"]))
        if state.get("summary"):
            prefix.append(
                SystemMessage(content="Conversation so far:\n" + state["summary"])
            )
        # Past the tool budget the unbound model runs, so a plain-text reply —
        # and therefore END — is guaranteed.
        active = model if state.get("tools_exhausted") else bound_model
        reply = active.invoke(prefix + list(state["messages"]))
        return {"messages": [reply]}

    def run_tools(state: BrainState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        rounds = state.get("tool_rounds", 0)
        if rounds >= settings.tool_max_rounds:
            return {
                "messages": [
                    ToolMessage(
                        content="Tool budget exhausted; answer the user with what you have.",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                    for call in calls
                ],
                "tools_exhausted": True,
            }
        ctx = ToolContext(
            settings=settings,
            thread_id=config.get("configurable", {}).get("thread_id", ""),
            batch_id=state.get("batch_id", ""),
        )
        results: list[BaseMessage] = []
        for call in calls:
            spec = by_name.get(call["name"])
            if spec is None:
                output = f"Unknown tool: {call['name']}. Available: {', '.join(by_name)}."
            else:
                output = execute_tool(spec, ctx, call.get("args") or {})
            results.append(
                ToolMessage(content=output, tool_call_id=call["id"], name=call["name"])
            )
        return {"messages": results, "tool_rounds": rounds + 1}

    def route_after_agent(state: BrainState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(BrainState)
    graph.add_node("recall", recall)
    graph.add_node("agenda", agenda)
    graph.add_node("tasks", tasks)
    graph.add_node("profile", profile)
    graph.add_node("agent", call_agent)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "agenda")
    graph.add_edge("agenda", "tasks")
    graph.add_edge("tasks", "profile")
    graph.add_edge("profile", "agent")
    if specs:
        graph.add_node("tools", run_tools)
        graph.add_conditional_edges("agent", route_after_agent, ["tools", END])
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)
    return graph.compile(checkpointer=_checkpointer(settings))
