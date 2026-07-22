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
from pydantic import ValidationError

from assistant.config import Settings
from assistant.docs import store, summarize
from assistant.docs.context import docs_context


def _embed_with_dim(dim: int):
    """The same deterministic bag-of-words embedder, at a chosen dimension."""

    def _embed(texts, prefix: str = "", settings=None):
        vecs = []
        for text in texts:
            v = [0.0] * dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[zlib.crc32(word.encode()) % dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs

    return _embed


_fake_embed = _embed_with_dim(64)


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


def test_chunk_survives_a_zero_target() -> None:
    # A zero target used to make the hard-split loop slice nothing off each pass,
    # so it spun forever. Reaching the assertions at all is the point.
    chunks = store.chunk_text("hello world", target_chars=0)
    assert "".join(chunks) == "hello world"
    assert all(chunks)  # no empty pieces


def test_zero_chunk_size_is_rejected_by_config() -> None:
    with pytest.raises(ValidationError):
        Settings(docs_chunk_chars=0)


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


# --- reindex / embedding-model migration ------------------------------------ #


def test_reindex_rebuilds_after_an_embedding_model_change(tmp_path, monkeypatch) -> None:
    memory_dir = str(tmp_path / "memory")

    def _at(model: str, dim: int) -> Settings:
        monkeypatch.setattr("assistant.memory.embeddings._embed", _embed_with_dim(dim))
        return Settings(
            memory_dir=memory_dir, embedding_model=model, docs_min_similarity=0.1
        )

    old = _at("fake/dim-64", 64)
    store.add_document(old, "Bergen", "The Bergen fish market sells salmon.")
    assert store.search_chunks(old, "fish market salmon")

    # Swap to a model of a different dimension. Without a migration the vec table
    # stays float[64] and every read and write raises "Dimension mismatch".
    new = _at("fake/dim-32", 32)
    assert store.reindex(new) == 1

    hits = store.search_chunks(new, "fish market salmon")
    assert hits and hits[0].doc_title == "Bergen"
    store.add_document(new, "Cars", "The engine needs an oil change.")  # writes work again


def test_reindex_embeds_nothing_when_the_model_is_unchanged(settings, monkeypatch) -> None:
    store.add_document(settings, "Bergen", "The Bergen fish market sells salmon.")

    embedded: list[int] = []

    def counting(texts, prefix: str = "", settings=None):
        embedded.append(len(texts))
        return _fake_embed(texts, prefix, settings)

    monkeypatch.setattr("assistant.memory.embeddings._embed", counting)
    assert store.reindex(settings) == 1
    assert embedded == []  # a routine restart must not re-embed the corpus


def test_reindex_migrates_a_pre_signature_db(settings, monkeypatch) -> None:
    # A db written before embedding_signature() stamped the *bare* model name.
    # The fastembed pooling swap keeps that name identical, so keying the
    # migration off the name alone would leave the corpus on stale vectors. The
    # signature must read the legacy stamp as changed and re-embed from source.
    store.add_document(settings, "Bergen", "The Bergen fish market sells salmon.")
    conn = store._connect(settings)
    try:
        store._meta_set(conn, "embedding_model", settings.embedding_model)  # legacy bare name
        conn.commit()
    finally:
        conn.close()

    embedded: list[int] = []

    def counting(texts, prefix: str = "", settings=None):
        embedded.append(len(texts))
        return _fake_embed(texts, prefix, settings)

    monkeypatch.setattr("assistant.memory.embeddings._embed", counting)
    assert store.reindex(settings) == 1
    assert embedded  # the corpus was re-embedded, not left on stale vectors
    hits = store.search_chunks(settings, "fish market salmon")
    assert hits and hits[0].doc_title == "Bergen"


# --- summarize -------------------------------------------------------------- #


def test_summarize_uses_model(settings, monkeypatch) -> None:
    doc = store.add_document(settings, "Report", "Q3 revenue rose 10%.")

    def fake_complete(prompt, s=None, **kwargs):
        assert "Q3 revenue" in prompt
        return "Revenue up 10% in Q3."

    monkeypatch.setattr(summarize, "complete_text", fake_complete)
    assert summarize.summarize_document(settings, doc.id) == "Revenue up 10% in Q3."


def test_summarize_missing_document_returns_none(settings) -> None:
    assert summarize.summarize_document(settings, "nope") is None


class _CountingCompleter:
    """Records every prompt it is asked to complete."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt, settings=None, **kwargs) -> str:
        self.prompts.append(prompt)
        return f"summary {len(self.prompts)}"


def test_short_document_summarizes_in_one_call(settings, monkeypatch) -> None:
    doc = store.add_document(settings, "Report", "Q3 revenue rose 10%.")
    completer = _CountingCompleter()
    monkeypatch.setattr(summarize, "complete_text", completer)

    assert summarize.summarize_document(settings, doc.id) == "summary 1"
    assert len(completer.prompts) == 1


def test_long_document_is_summarized_map_reduce(tmp_path, monkeypatch) -> None:
    # A target small enough that the document spans several pieces; a single
    # prompt carrying the whole text would overflow a real model's context.
    s = Settings(memory_dir=str(tmp_path / "m"), docs_summarize_chars=20)
    doc = store.add_document(
        s, "Report", "Q3 revenue rose.\n\nQ4 revenue fell.\n\nHiring is paused."
    )
    completer = _CountingCompleter()
    monkeypatch.setattr(summarize, "complete_text", completer)

    result = summarize.summarize_document(s, doc.id)

    assert len(completer.prompts) > 2  # one per section (map) plus the final fold
    assert all("one section" in p for p in completer.prompts[:-1])
    assert "Section summaries:" in completer.prompts[-1]
    assert result == f"summary {len(completer.prompts)}"  # the reduce output wins
