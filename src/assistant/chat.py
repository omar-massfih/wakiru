"""Channel-agnostic chat core — one conversation turn plus its upkeep.

Every channel (the HTTP API, Telegram) speaks to the agent the same way: invoke
the graph for a reply, then run the post-reply upkeep — long-term memory,
working-memory folding, calendar extraction, periodic consolidation — *after*
the reply has been delivered. Centralizing it here keeps channel behavior
identical and the channels themselves thin.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from .agent import maybe_summarize
from .calendar import update_calendar
from .config import Settings, get_settings
from .memory import consolidate_memory, index, update_memory
from .tasks import update_tasks
from .undo import undo_latest

logger = logging.getLogger(__name__)


def _is_undo_command(settings: Settings, message: str) -> bool:
    """Whether ``message`` asks to undo the last calendar or task write on this thread."""
    if not settings.enable_write_confirmation:
        return False
    if not (settings.enable_calendar or settings.enable_tasks):
        return False
    text = message.strip().lower()
    return text == "undo" or text.startswith("undo ")


def run_chat(
    agent: CompiledStateGraph,
    message: str,
    thread_id: str,
    settings: Settings | None = None,
) -> str:
    """Run one turn on ``thread_id`` and return the reply.

    A message that asks to undo the last calendar write ("undo", "undo that", …)
    is resolved deterministically against the undo ledger and short-circuits the
    LLM entirely — no ambiguity, no added latency, matching :func:`run_upkeep`'s
    equivalent skip so the turn is never double-processed.

    Raises :class:`assistant.codex_runner.CodexError` when the model fails; each
    channel translates that into its own error surface (HTTP 502, a chat apology).
    """
    settings = settings or get_settings()
    if _is_undo_command(settings, message):
        return undo_latest(settings, thread_id, settings.write_undo_window_minutes)

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    reply = result["messages"][-1].content
    return reply if isinstance(reply, str) else str(reply)


async def run_chat_stream(
    agent: CompiledStateGraph,
    message: str,
    thread_id: str,
    settings: Settings | None = None,
) -> AsyncIterator[str]:
    """Run one turn on ``thread_id``, yielding the reply incrementally.

    Yields text chunks as the model produces them (via ``agent.astream`` in
    ``"messages"`` mode). Providers that can't stream token-by-token — the Codex
    CLI, whose subprocess only returns a finished message — still work: LangGraph
    emits the whole reply as a single chunk, so a consumer sees one yield.

    An "undo" command short-circuits exactly like :func:`run_chat` and yields the
    deterministic ledger result as a single chunk. The caller is responsible for
    running :func:`run_upkeep` once the stream is exhausted.

    Raises :class:`assistant.codex_runner.CodexError` when the model fails.
    """
    settings = settings or get_settings()
    if _is_undo_command(settings, message):
        yield undo_latest(settings, thread_id, settings.write_undo_window_minutes)
        return

    config = {"configurable": {"thread_id": thread_id}}
    async for chunk, _meta in agent.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="messages",
    ):
        # Only the codex node's model output is user-facing; other nodes (recall,
        # agenda) don't emit message chunks in this mode. Skip non-AI chunks and
        # empty deltas so the consumer sees only reply text.
        if isinstance(chunk, AIMessageChunk):
            text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if text:
                yield text


def run_upkeep(
    agent: CompiledStateGraph,
    settings: Settings,
    message: str,
    reply: str,
    thread_id: str,
) -> None:
    """All post-reply maintenance for one turn, best-effort piece by piece.

    Meant to run off the reply path (a FastAPI background task, a channel worker
    thread) so it never adds latency. Each step is isolated so a failure in one
    never starves the others.
    """
    if _is_undo_command(settings, message):
        # Handled synchronously in run_chat; nothing here to learn or extract —
        # and re-running the calendar extractor on "undo" text risks it
        # hallucinating its own op from the reply that already reported the undo.
        return

    # Long-term memory: an episodic trace + a reconciling save/update/forget pass.
    try:
        update_memory(settings, message, reply, thread_id)
    except Exception:
        logger.exception("memory upkeep failed for thread %s", thread_id)

    # Working memory: fold older turns into the rolling summary past the threshold.
    try:
        maybe_summarize(agent, settings, thread_id)
    except Exception:
        logger.exception("summarization upkeep failed for thread %s", thread_id)

    # Calendar: a reconciling extraction that creates/reschedules/cancels events.
    if settings.enable_calendar and settings.enable_auto_schedule:
        try:
            update_calendar(settings, message, reply, thread_id)
        except Exception:
            logger.exception("calendar upkeep failed for thread %s", thread_id)

    # Tasks: a reconciling extraction that adds/completes/updates/removes to-dos.
    if settings.enable_tasks and settings.enable_auto_tasks:
        try:
            update_tasks(settings, message, reply, thread_id)
        except Exception:
            logger.exception("task upkeep failed for thread %s", thread_id)

    # Periodic consolidation ("sleep"); the counter persists across restarts.
    try:
        every = settings.consolidate_every_n_turns
        if every > 0 and index.bump_turn_counter(settings) % every == 0:
            consolidate_memory(settings)
    except Exception:
        logger.exception("consolidation failed")
