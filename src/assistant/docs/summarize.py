"""Summarize an ingested document with the configured chat model."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from ..config import Settings, get_settings
from ..llm import build_model
from . import store

_PROMPT = (
    "Summarize the following document concisely, preserving its key points, "
    "decisions, and any action items. Use short paragraphs or bullet points.\n\n"
    "Title: {title}\n\n{text}"
)


def summarize_document(settings: Settings | None, doc_id: str) -> str | None:
    """Return a summary of the stored document, or ``None`` if it doesn't exist.

    Runs one model call via the configured provider (:func:`assistant.llm.build_model`),
    so it works with codex or an API-backed provider alike.
    """
    settings = settings or get_settings()
    doc = store.get_document(settings, doc_id)
    if doc is None:
        return None
    model = build_model(settings)
    prompt = _PROMPT.format(title=doc.title, text=doc.text)
    reply = model.invoke([HumanMessage(content=prompt)]).content
    return reply if isinstance(reply, str) else str(reply)
