"""Memory tests — store round-trip, vector index, dedup, recall, forget.

Embeddings are faked with a deterministic bag-of-words vector so these stay fast
and offline; only the sqlite-vec index and the file store run for real.
"""

from __future__ import annotations

import math
import re

import pytest

from assistant.config import Settings
from assistant.memory import learn, recall, store
from assistant.memory.store import Note


def _fake_embed(texts: list[str], settings=None) -> list[list[float]]:
    """Normalized term-frequency vectors: overlap in words -> high cosine."""
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[hash(word) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=False)


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    monkeypatch.setattr("assistant.memory.embeddings.embed", _fake_embed)


# --- store ---------------------------------------------------------------- #


def test_note_roundtrip(settings) -> None:
    note = Note(name="user-name", description="The user is Omar", body="The user's name is Omar.")
    path = store.write_note(settings, note)
    assert path.parent.name == "facts"  # fact -> facts/
    back = store.read_note(path)
    assert back.name == "user-name"
    assert back.description == "The user is Omar"
    assert back.body == "The user's name is Omar."
    assert back.type == "fact"


def test_learning_goes_to_learnings_dir(settings) -> None:
    note = Note(name="deploy", description="deploy uses uv", body="Deploy with uv.", type="learning")
    path = store.write_note(settings, note)
    assert path.parent.name == "learnings"


def test_regenerate_index_lists_notes(settings) -> None:
    store.write_note(settings, Note(name="a", description="first fact", body="A."))
    store.write_note(settings, Note(name="b", description="second fact", body="B."))
    store.regenerate_index(settings)
    text = store.read_index(settings)
    assert "first fact" in text and "second fact" in text
    assert "**a**" in text and "**b**" in text


# --- index + recall ------------------------------------------------------- #


def test_save_and_recall(settings) -> None:
    # Vocabulary overlaps on purpose: the fake embedder scores by shared words,
    # so this exercises index + threshold + note-loading, not semantics.
    learn.save_memory(settings, body="The user prefers Norwegian language replies.")
    results = recall.search_memory(settings, "Norwegian language replies preference")
    assert results, "expected the Norwegian preference to be recalled"
    assert "norwegian" in results[0][0].body.lower()


def test_recall_below_threshold_returns_nothing(settings) -> None:
    learn.save_memory(settings, body="The user prefers replies in Norwegian.")
    # Completely unrelated query -> no word overlap -> filtered out.
    assert recall.search_memory(settings, "quarterly budget spreadsheet totals") == []


def test_dedup_updates_in_place(settings) -> None:
    learn.save_memory(settings, body="The user prefers replies in Norwegian.")
    learn.save_memory(settings, body="The user prefers replies in Norwegian please.")
    notes = store.list_notes(settings)
    assert len(notes) == 1, "a near-duplicate should update, not create a second note"


# --- forget --------------------------------------------------------------- #


def test_forget_memory_deletes_best_match(settings) -> None:
    learn.save_memory(settings, body="The user's favorite color is teal.")
    deleted = learn.forget_memory(settings, "favorite color teal")
    assert deleted is not None
    assert store.list_notes(settings) == []


def test_forget_memory_no_match_returns_none(settings) -> None:
    learn.save_memory(settings, body="The user's favorite color is teal.")
    assert learn.forget_memory(settings, "quarterly budget spreadsheet") is None
    assert store.list_notes(settings), "an unrelated forget must not delete anything"


# --- LLM-driven update (save + forget in one pass) ------------------------ #


def test_update_memory_applies_save_and_forget_ops(tmp_path, monkeypatch) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=True)
    learn.save_memory(settings, body="The user's favorite color is teal.")

    canned = (
        '[{"op": "save", "type": "fact", "description": "Lives in Oslo",'
        ' "body": "The user lives in Oslo."},'
        ' {"op": "forget", "query": "favorite color teal"}]'
    )
    monkeypatch.setattr("assistant.memory.learn.run_codex", lambda *a, **k: canned)

    applied = learn.update_memory(settings, "user text", "assistant text")

    names = {n.name for n in store.list_notes(settings)}
    assert "lives-in-oslo" in names, "save op should create the Oslo note"
    assert all("teal" not in n.body.lower() for n in store.list_notes(settings)), (
        "forget op should delete the teal note"
    )
    assert any(s.startswith("saved:") for s in applied)
    assert any(s.startswith("forgot:") for s in applied)


def test_update_memory_disabled_is_noop(tmp_path, monkeypatch) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=False)
    monkeypatch.setattr(
        "assistant.memory.learn.run_codex",
        lambda *a, **k: pytest.fail("run_codex must not be called when disabled"),
    )
    assert learn.update_memory(settings, "hi", "hello") == []
