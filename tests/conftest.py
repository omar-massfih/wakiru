"""Test isolation from the developer's real configuration.

``Settings`` reads ``.env`` and the ambient environment by design — which means a
bare ``Settings()`` in a test would otherwise pick up the *real* bot token, API
key, or mailbox credentials of whoever runs the suite. Every test that touches a
transport monkeypatches it, so nothing has escaped so far, but that is a property
of each test rather than of the harness: one forgotten patch and the suite talks
to a live Telegram bot or mailbox.

These autouse fixtures make the isolation structural. Tests see a Settings whose
every field is at its declared default unless the test passes it explicitly.
"""

from __future__ import annotations

import math
import os
import re
import zlib

import pytest

from assistant.config import Settings, get_settings


def _bag_of_words_embed(texts, prefix: str = "", settings=None) -> list[list[float]]:
    """Deterministic, offline stand-in for the real embedder.

    Normalized bag-of-words vectors — word overlap yields high cosine — so the
    sqlite-vec / pgvector index runs for real while nothing loads the actual
    ~2GB fastembed model. Matches the per-file fakes in test_memory.py and
    test_docs.py (64-dim, crc32 word hashing) so behaviour is identical whether a
    test patches ``_embed`` itself or leans on the session default below.
    """
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch) -> None:
    """Route every test's embeddings through the deterministic fake by default.

    ``embeddings._embed`` is the single seam every embed wrapper funnels through,
    so patching it here keeps the whole suite offline and fast — a test that
    reaches recall/docs without faking embeddings itself no longer silently loads
    the real model. Tests that need their own fake still override this (function
    scope, same seam). Set ``REAL_EMBEDDINGS=1`` to exercise the genuine model —
    what test_recall_real.py asserts — which opts the whole run into real vectors.
    """
    if os.environ.get("REAL_EMBEDDINGS"):
        return
    monkeypatch.setattr("assistant.memory.embeddings._embed", _bag_of_words_embed)

# Field names whose values are credentials or would reach the network. Cleared
# from os.environ for the whole session; the rest of the fields are harmless.
_SENSITIVE_ENV = [
    name.upper()
    for name in Settings.model_fields
    if any(
        token in name
        for token in ("token", "key", "secret", "password", "url", "address", "chat_ids", "user_ids")
    )
]


@pytest.fixture(autouse=True, scope="session")
def _ignore_dotenv() -> None:
    """Stop pydantic-settings from loading the developer's .env during tests."""
    Settings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch) -> None:
    """Clear credential-ish env vars so an exported one can't leak into a test."""
    for name in _SENSITIVE_ENV:
        monkeypatch.delenv(name, raising=False)
    # The default quiet window would make reminder/briefing/heartbeat tests
    # pass or fail depending on the wall clock they run at. Disable it for the
    # suite; tests exercising the default set quiet_hours_default explicitly.
    monkeypatch.setenv("QUIET_HOURS_DEFAULT", "")
    # get_settings() is lru_cached; a cached instance built from the real
    # environment would survive the patches above.
    get_settings.cache_clear()

