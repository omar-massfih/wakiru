"""Summarize an ingested document with the configured chat model.

A document can be arbitrarily long — longer than any model's context — so a
single prompt carrying its whole text is not safe. Long documents are summarized
map-reduce style: each piece is summarized on its own (the *map*), then those
partial summaries are folded into one (the *reduce*). A document that already
fits in one piece takes the single-call path, exactly as before.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..llm import complete_text
from . import store

_PROMPT = (
    "Summarize the following document concisely, preserving its key points, "
    "decisions, and any action items. Use short paragraphs or bullet points.\n\n"
    "Title: {title}\n\n{text}"
)

# Map step: one section of a long document. Kept factual and lossless-ish, since
# the reduce step below is what does the final compression.
_MAP_PROMPT = (
    "The following is one section of a longer document. Summarize just this "
    "section, preserving its key points, decisions, and any action items. Do not "
    "speculate about the rest of the document.\n\n"
    "Title: {title}\n\n{text}"
)

# Reduce step: fold the per-section summaries into one document-level summary.
_REDUCE_PROMPT = (
    "The following are summaries of consecutive sections of one document, in "
    "order. Combine them into a single coherent summary of the whole document, "
    "preserving its key points, decisions, and any action items. Merge "
    "repetition; do not mention that it was summarized in sections. Use short "
    "paragraphs or bullet points.\n\n"
    "Title: {title}\n\nSection summaries:\n\n{text}"
)


def _invoke(settings: Settings, prompt: str, title: str, text: str) -> str:
    return complete_text(prompt.format(title=title, text=text), settings)


def summarize_document(settings: Settings | None, doc_id: str) -> str | None:
    """Return a summary of the stored document, or ``None`` if it doesn't exist.

    Runs via the configured provider (:func:`assistant.llm.complete_text`), so
    it works with codex or an API-backed provider alike. A document that fits in
    one ``docs_summarize_chars`` piece costs one model call; a longer one costs
    one call per piece plus a final fold.
    """
    settings = settings or get_settings()
    doc = store.get_document(settings, doc_id)
    if doc is None:
        return None

    pieces = store.chunk_text(doc.text, settings.docs_summarize_chars)
    if len(pieces) <= 1:
        return _invoke(settings, _PROMPT, doc.title, doc.text)

    partials = [_invoke(settings, _MAP_PROMPT, doc.title, piece) for piece in pieces]
    return _invoke(settings, _REDUCE_PROMPT, doc.title, "\n\n".join(partials))
