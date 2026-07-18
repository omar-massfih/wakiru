"""Channel-agnostic chat core — one conversation turn plus its upkeep.

Every channel (the HTTP API, Telegram) speaks to the agent the same way: invoke
the graph for a reply, then run the post-reply upkeep — long-term memory,
working-memory folding, periodic consolidation — *after* the reply has been
delivered. Centralizing it here keeps channel behavior identical and the
channels themselves thin.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from . import threads
from .agent import maybe_summarize
from .chatgpt_backend import ChatGptAuthError, ChatGptError, ChatGptTimeoutError
from .codex_runner import CodexError, CodexTimeoutError
from .config import Settings, get_settings
from .memory import consolidate_memory, index, update_memory

logger = logging.getLogger(__name__)


def error_reply(exc: Exception) -> str:
    """A human explanation of a failed turn, by failure kind.

    Chat channels (Telegram, Slack) show this instead of a one-size apology —
    or worse, silence. Deliberately content-free about internals: the log has
    the traceback, the user just needs to know whether retrying can help.
    """
    if isinstance(exc, CodexTimeoutError | ChatGptTimeoutError | TimeoutError):
        return (
            "That one took too long and I gave up partway. "
            "Try again — or break it into smaller steps."
        )
    if isinstance(exc, ChatGptAuthError):
        # User-actionable, unlike a transient snag: the ChatGPT sign-in the
        # chatgpt provider borrows has expired or gone missing.
        return (
            "I can't reach ChatGPT right now — my sign-in looks expired. "
            "Run `codex login` on the server, then try again."
        )
    if isinstance(exc, CodexError | ChatGptError):
        return "My reasoning engine hit a snag. Give it a moment and try again."
    return "Something unexpected broke on my end — it's logged. Try once more."


def run_chat(
    agent: CompiledStateGraph,
    message: str,
    thread_id: str,
    settings: Settings | None = None,
) -> str:
    """Run one turn on ``thread_id`` and return the reply.

    Every message is the model's to interpret — including "undo", which it
    resolves by calling the ``undo`` tool against the ledger.

    Raises the provider's error (:class:`assistant.codex_runner.CodexError`,
    :class:`assistant.chatgpt_backend.ChatGptError`, …) when the model fails;
    each channel translates that into its own error surface (HTTP 502, a chat
    apology).
    """
    settings = settings or get_settings()
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
    ``"messages"`` mode). Every provider streams: the Codex provider parses the
    CLI's ``--json`` event stream (:func:`assistant.codex_runner.run_codex_stream`);
    worst case a provider emits the whole reply as one chunk.

    The caller is responsible for running :func:`run_upkeep` once the stream
    is exhausted.

    Raises :class:`assistant.codex_runner.CodexError` when the model fails.
    """
    settings = settings or get_settings()
    config = {"configurable": {"thread_id": thread_id}}
    async for chunk, _meta in agent.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="messages",
    ):
        # Only the agent node's model output is user-facing; other nodes (recall,
        # agenda) don't emit message chunks in this mode. Skip non-AI chunks,
        # tool-call chunks (structured intent, not reply text — the codex shim
        # withholds them and the native providers emit them content-free), and
        # empty deltas so the consumer sees only reply text.
        if isinstance(chunk, AIMessageChunk):
            if chunk.tool_call_chunks or chunk.tool_calls:
                continue
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
    # Thread registry: every channel funnels through here, so this one touch
    # keeps the registry of live conversations current (Slack loop-in, and the
    # heartbeat's "time since last contact").
    try:
        threads.touch(settings, thread_id)
    except Exception:
        logger.exception("thread registry touch failed for thread %s", thread_id)

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

    # Periodic consolidation ("sleep"); the counter persists across restarts.
    try:
        every = settings.consolidate_every_n_turns
        if every > 0 and index.bump_turn_counter(settings) % every == 0:
            consolidate_memory(settings)
    except Exception:
        logger.exception("consolidation failed")
