"""Document subsystem tests — ingest, chunk, recall, summarize.

Embeddings are faked with the same deterministic bag-of-words vector as
test_memory.py (patched at ``embeddings._embed``, the single seam every embed
wrapper funnels through), so the sqlite-vec index runs for real while staying
fast and offline.
"""

from __future__ import annotations

import math
import re
import zlib

import pytest

from assistant.config import Settings
from assistant.docs import store, summarize
from assistant.docs.context import docs_context


def _fake_embed(texts, prefix: str = "", settings=None):
    vecs = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    monkeypatch.setattr("assistant.memory.embeddings._embed", _fake_embed)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), docs_min_similarity=0.1)


# --- chunking --------------------------------------------------------------- #


def test_chunk_splits_on_paragraphs() -> None:
    text = "para one here\n\npara two here\n\npara three here"
    chunks = store.chunk_text(text, target_chars=20)
    assert len(chunks) == 3
    assert chunks[0] == "para one here"


def test_chunk_packs_small_paragraphs() -> None:
    text = "a\n\nb\n\nc"
    assert store.chunk_text(text, target_chars=1000) == ["a\n\nb\n\nc"]


def test_chunk_hard_splits_a_huge_paragraph() -> None:
    chunks = store.chunk_text("x" * 500, target_chars=100)
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks) == "x" * 500


# --- ingest + list + delete ------------------------------------------------- #


def test_add_document_indexes_chunks(settings) -> None:
    # Short paragraphs pack into a single chunk at the default 800-char target.
    doc = store.add_document(
        settings, "Trip notes", "We visited Bergen.\n\nThe fish market was great."
    )
    assert doc.chunks == 1
    listed = store.list_documents(settings)
    assert len(listed) == 1 and listed[0].title == "Trip notes" and listed[0].chunks == 1


def test_long_document_indexes_multiple_chunks(tmp_path) -> None:
    settings = Settings(memory_dir=str(tmp_path / "m"), docs_chunk_chars=20)
    doc = store.add_document(
        settings, "Trip notes", "We visited Bergen.\n\nThe fish market was great."
    )
    assert doc.chunks == 2
    assert store.list_documents(settings)[0].chunks == 2


def test_delete_document_removes_it(settings) -> None:
    doc = store.add_document(settings, "T", "some content here")
    assert store.delete_document(settings, doc.id) is True
    assert store.list_documents(settings) == []
    assert store.delete_document(settings, doc.id) is False  # already gone


# --- recall ----------------------------------------------------------------- #


def test_search_finds_relevant_chunk(settings) -> None:
    store.add_document(settings, "Bergen", "The Bergen fish market sells salmon.")
    store.add_document(settings, "Cars", "The engine needs an oil change.")
    hits = store.search_chunks(settings, "fish market salmon")
    assert hits and hits[0].doc_title == "Bergen"
    assert "fish market" in hits[0].text


def test_search_empty_when_nothing_ingested(settings) -> None:
    assert store.search_chunks(settings, "anything") == []


def test_docs_context_renders_matches(settings) -> None:
    store.add_document(settings, "Bergen", "The Bergen fish market sells salmon.")
    block = docs_context(settings, "salmon fish market")
    assert "Relevant excerpts" in block and "Bergen" in block


def test_docs_context_disabled_returns_empty(tmp_path) -> None:
    s = Settings(memory_dir=str(tmp_path / "m"), enable_docs=False)
    store.add_document(s, "x", "content")
    assert docs_context(s, "content") == ""


# --- summarize -------------------------------------------------------------- #


def test_summarize_uses_model(settings, monkeypatch) -> None:
    doc = store.add_document(settings, "Report", "Q3 revenue rose 10%.")

    class _FakeModel:
        def invoke(self, messages):
            assert "Q3 revenue" in messages[0].content
            return type("Msg", (), {"content": "Revenue up 10% in Q3."})()

    monkeypatch.setattr(summarize, "build_model", lambda s: _FakeModel())
    assert summarize.summarize_document(settings, doc.id) == "Revenue up 10% in Q3."


def test_summarize_missing_document_returns_none(settings) -> None:
    assert summarize.summarize_document(settings, "nope") is None
