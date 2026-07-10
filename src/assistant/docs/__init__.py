"""Document ingest + recall + summarization.

Ingested documents are chunked and embedded into their own ``docs.db`` vector
index (mirroring the memory brain's machinery), so their most relevant chunks can
be recalled semantically each turn (:func:`docs_context`, wired into the graph's
``recall`` node) and a whole document can be summarized on demand
(:func:`summarize_document`). Kept separate from the memory brain so document
chunks never dilute durable notes.
"""

from __future__ import annotations

from . import store
from .context import docs_context
from .store import Chunk, Document, add_document, search_chunks
from .summarize import summarize_document

__all__ = [
    "Chunk",
    "Document",
    "add_document",
    "docs_context",
    "search_chunks",
    "store",
    "summarize_document",
]
