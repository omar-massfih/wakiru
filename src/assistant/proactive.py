"""Proactive loop-in — connect unprompted pushes back to the conversation.

Reminders, the briefing, and heartbeat messages are pushed by background
paths, outside any conversation. Historically they were fire-and-forget: the
push landed on the user's phone but the assistant had no record of it, so
"what was that reminder about?" drew a blank. :func:`record_push` closes the
loop: it appends what was delivered to each target conversation's working
memory (as an ``AIMessage``, exactly as the user saw it), so the next turn on
that thread knows about it and can follow up.

Nothing here binds tools; the heartbeat (:mod:`assistant.heartbeat`) binds
the restricted ``mode="heartbeat"`` registry, which can never contain
``send_email`` — either way, no background path can send mail.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from .config import Settings

logger = logging.getLogger(__name__)


def target_threads(settings: Settings) -> list[str]:
    """The conversation threads proactive pushes should be recorded on.

    Telegram chats each map to a stable thread (``telegram:<chat_id>``), so the
    authorized set is the delivery set. Slack thread ids embed the *user* who
    spoke (``slack:<channel>:<user>``), which a broadcast can't reconstruct —
    but the thread registry (:mod:`assistant.threads`) knows every Slack thread
    that has actually talked to the assistant, and a push lands in
    ``slack_notify_channel``; recording it on the registered threads *in that
    channel* records it exactly where it was seen.
    """
    targets: list[str] = []
    if settings.telegram_bot_token:
        from .telegram import authorized_chats

        targets += [f"telegram:{chat_id}" for chat_id in authorized_chats(settings)]
    if settings.slack_bot_token and settings.slack_notify_channel:
        from . import threads

        try:
            for info in threads.known_threads(settings, channel="slack"):
                # slack:<channel>:<user> — only threads living in the channel
                # the push was delivered to actually saw it.
                parts = info.thread_id.split(":", 2)
                if len(parts) == 3 and parts[1] == settings.slack_notify_channel:
                    targets.append(info.thread_id)
        except Exception:
            logger.exception("slack thread lookup failed; recording to telegram only")
    return targets


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


