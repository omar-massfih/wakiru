"""Local, offline text embeddings via fastembed (ONNX — no torch, no API key).

The default model is ``intfloat/multilingual-e5-large`` — strong multilingual
retrieval (Norwegian included), 1024-dim. The model is loaded lazily and cached
process-wide; the first call downloads it into the HuggingFace cache once (~2GB),
every call afterwards runs fully offline.

**e5 is asymmetric.** It expects stored documents to be prefixed with
``"passage: "`` and search queries with ``"query: "``. Mixing these up quietly
wrecks recall, so callers should go through :func:`embed_passages` (for stored
notes) and :func:`embed_query` (for lookups) rather than prefixing by hand. For
symmetric models (no ``e5`` in the name) the prefixes are omitted automatically.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import Settings, get_settings

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


@lru_cache(maxsize=4)
def _embedder(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _embed(texts: list[str], prefix: str, settings: Settings | None) -> list[list[float]]:
    settings = settings or get_settings()
    if not texts:
        return []
    model_name = settings.embedding_model
    pre = prefix if _is_e5(model_name) else ""
    model = _embedder(model_name)
    return [vec.tolist() for vec in model.embed([pre + t for t in texts])]


def embed_passages(
    texts: list[str], settings: Settings | None = None
) -> list[list[float]]:
    """Embed texts as stored documents (``passage:`` side of an e5 model)."""
    return _embed(texts, _PASSAGE_PREFIX, settings)


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a single search query (``query:`` side of an e5 model)."""
    return _embed([text], _QUERY_PREFIX, settings)[0]


def embed_one(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a single stored text (passage side)."""
    return embed_passages([text], settings)[0]
