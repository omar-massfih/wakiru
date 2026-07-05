"""Real-embedder recall test — proves genuine *semantic* recall.

Unlike test_memory.py (which fakes embeddings to test plumbing), this loads the
actual local fastembed model. It needs the model in the HuggingFace cache; the
first run downloads it, every run afterwards is offline.
"""

from __future__ import annotations

from assistant.config import Settings
from assistant.memory import learn, recall


def test_semantic_recall_no_shared_words(tmp_path) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=False)
    learn.save_memory(settings, body="The user prefers replies in Norwegian.")

    # Note the query shares essentially no words with the stored note — only a
    # real embedding model can bridge "language/answer" to "Norwegian/replies".
    results = recall.search_memory(settings, "Which language should you answer me in?")

    assert results, "semantic recall should surface the Norwegian preference"
    assert "norwegian" in results[0][0].body.lower()
