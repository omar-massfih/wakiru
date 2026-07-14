"""One seam for speaking as Wakiru in the background — with memory, no tools.

Background pushes (reminder nudges, the briefing) should sound like the
assistant, not a template: the composer assembles the same prefix a chat turn
gets — persona charter, recalled memories, profile, agenda, tasks, mail —
and asks the model for one message. Unlike the heartbeat's deliberative wake
it binds no tools: the reflex paths that call this must stay bounded, and
composition is the only job.

Failure never loses a message: any exception, timeout, or empty reply falls
back to the caller-supplied deterministic text (the :mod:`assistant.phrasing`
template, the verbatim digest), so the push goes out either way.
"""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from . import persona
from .config import Settings
from .context_providers import build_context
from .llm import build_model

logger = logging.getLogger(__name__)


def compose_push(
    settings: Settings,
    *,
    instruction: str,
    facts: str,
    query: str,
    fallback: str,
) -> str:
    """Compose one background push in the assistant's own voice.

    ``instruction`` says what kind of message to write, ``facts`` carries the
    deterministic source material (due reminders, briefing sections), and
    ``query`` drives memory recall so the message can lean on what the
    assistant knows. Returns ``fallback`` — logging, never raising — when the
    model fails or replies with nothing.
    """
    try:
        prefix: list[BaseMessage] = [persona.system_message(settings)]
        for _name, block in build_context(settings, query, "").items():
            if block:
                prefix.append(SystemMessage(content=block))
        prefix.append(SystemMessage(content=instruction))
        reply = build_model(settings).invoke(prefix + [HumanMessage(content=facts)])
        content = reply.content
        if not isinstance(content, str):
            # Anthropic can return a list of content blocks; keep the text ones.
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        text = content.strip()
        return text if text else fallback
    except Exception:
        logger.exception("background composition failed; using the fallback text")
        return fallback
