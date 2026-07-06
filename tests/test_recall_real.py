"""Real-embedder recall test — proves genuine, multilingual semantic recall.

Unlike test_memory.py (which fakes embeddings to test plumbing), this loads the
actual local fastembed model configured in Settings (multilingual e5-large by
default). It needs the model in the HuggingFace cache; the first run downloads it
(~2GB), every run afterwards is offline — so it is slow the first time.
"""

from __future__ import annotations

from assistant.config import Settings
from assistant.memory import learn, recall


def test_semantic_recall_no_shared_words(tmp_path) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=False)
    learn.save_memory(settings, body="The user prefers replies in Norwegian.")

    # The query shares essentially no words with the stored note — only a real
    # embedding model can bridge "language/answer" to "Norwegian/replies".
    results = recall.search_memory(settings, "Which language should you answer me in?")

    assert results, "semantic recall should surface the Norwegian preference"
    assert "norwegian" in results[0][0].body.lower()


def test_cross_lingual_recall(tmp_path) -> None:
    """A Norwegian memory should surface for an English query (multilingual)."""
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=False)
    learn.save_memory(settings, body="Brukeren bor i Bergen.")  # "The user lives in Bergen."

    results = recall.search_memory(settings, "Where does the user live?")

    assert results, "a Norwegian note should be recalled for an English query"
    assert "bergen" in results[0][0].body.lower()


def test_dedup_thresholds_with_real_model(tmp_path) -> None:
    """Distinct facts stay separate; a contradiction updates in place.

    Guards the e5-calibrated ``dedup_threshold`` (0.90): with the real model,
    "name is Omar" and "lives in Oslo" must NOT merge, but "lives in Bergen"
    should supersede "lives in Oslo".
    """
    from assistant.memory import store

    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=False)
    learn.save_memory(settings, body="The user's name is Omar.", kind="semantic")
    learn.save_memory(settings, body="The user lives in Oslo.", kind="semantic")
    assert len(store.list_notes(settings)) == 2, "distinct facts must not be merged"

    learn.save_memory(settings, body="The user lives in Bergen.", kind="semantic")
    notes = store.list_notes(settings)
    assert len(notes) == 2, "a contradiction should update in place, not add a third"
    bodies = " ".join(n.body.lower() for n in notes)
    assert "bergen" in bodies and "oslo" not in bodies
