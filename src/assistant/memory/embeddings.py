"""Local, offline text embeddings via fastembed (ONNX — no torch, no API key).

The model is loaded lazily and cached process-wide. The first call downloads the
model into the HuggingFace cache once; every call afterwards runs fully offline.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import Settings, get_settings


@lru_cache(maxsize=4)
def _embedder(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def embed(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Embed a batch of texts into normalized float vectors."""
    settings = settings or get_settings()
    if not texts:
        return []
    model = _embedder(settings.embedding_model)
    return [vec.tolist() for vec in model.embed(texts)]


def embed_one(text: str, settings: Settings | None = None) -> list[float]:
    return embed([text], settings=settings)[0]
