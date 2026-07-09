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

import pytest

from assistant.config import Settings, get_settings

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
    # get_settings() is lru_cached; a cached instance built from the real
    # environment would survive the patches above.
    get_settings.cache_clear()

