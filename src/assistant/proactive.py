"""Proactive loop-in — connect unprompted pushes back to the conversation.

Reminders and the daily briefing are computed by wall-clock tickers, outside
any conversation. Historically they were fire-and-forget: the push landed on
the user's phone but the assistant had no record of it, so "what was that
reminder about?" drew a blank. This module closes that loop two ways:

* :func:`record_push` appends what was delivered to each authorized chat's
  working memory (as an ``AIMessage``, exactly as the user saw it), so the
  next turn on that thread knows about it and can follow up.
* :func:`compose_briefing` writes the daily briefing *with* the user's profile
  context through the configured provider (replacing the old raw-codex polish
  pass), so the one proactive message that is prose reads like the assistant,
  not a template.

No tools are ever bound on these background paths — a proactive composition
must not be able to reach ``send_email`` or any other write.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from .config import Settings

logger = logging.getLogger(__name__)

_BRIEFING_INSTRUCTION = (
    "You are the user's personal assistant composing their daily morning "
    "briefing. Rewrite the digest below as a short, friendly briefing — a few "
    "sentences, plain text, no headings. Lead with what matters most today. "
    "Do not invent anything that is not in the digest. You cannot call tools "
    "or take actions here."
)


def target_threads(settings: Settings) -> list[str]:
    """The conversation threads proactive pushes should be recorded on.

    Telegram chats each map to a stable thread (``telegram:<chat_id>``). Slack
    thread ids embed the *user* who spoke (``slack:<channel>:<user>``), which a
    broadcast can't reconstruct — so Slack pushes are delivered but not
    recorded (a known v1 limitation).
    """
    if not settings.telegram_bot_token:
        return []
    from .telegram import authorized_chats

    return [f"telegram:{chat_id}" for chat_id in authorized_chats(settings)]


def record_to_thread(
    agent: CompiledStateGraph, settings: Settings, thread_id: str, text: str
) -> None:
    """Append a delivered push to one thread's working memory, best-effort."""
    try:
        agent.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [AIMessage(content=text)]},
            as_node="agent",
        )
    except Exception:
        logger.exception("failed to record push on thread %s", thread_id)


def record_push(
    agent: CompiledStateGraph | None, settings: Settings, text: str
) -> None:
    """Record a delivered push on every target thread (no-op without an agent)."""
    if agent is None or not settings.enable_proactive_loop_in or not text:
        return
    for thread_id in target_threads(settings):
        record_to_thread(agent, settings, thread_id, text)


def compose_briefing(settings: Settings, digest: str) -> str:
    """One profile-aware LLM pass over the digest; the raw digest is the fallback."""
    from .llm import complete_text
    from .memory.profile import profile_context

    system = _BRIEFING_INSTRUCTION
    try:
        profile = profile_context(settings)
    except Exception:
        logger.exception("briefing: profile context failed; composing without it")
        profile = ""
    if profile:
        system += "\n\n" + profile

    try:
        reply = complete_text(digest, settings, system=system)
    except Exception:
        logger.exception("briefing composition failed; sending the raw digest")
        return digest
    return reply.strip() or digest
