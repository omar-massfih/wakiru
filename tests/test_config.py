"""Configuration tests — chiefly, that tests never see the real environment.

``Settings`` reads ``.env`` and os.environ by design, so without the isolation in
conftest.py a bare ``Settings()`` here would pick up the developer's live bot
token, API key, and mailbox credentials. This asserts the isolation holds: if it
ever regresses, a forgotten monkeypatch elsewhere could reach a live service.
"""

from __future__ import annotations

from assistant.config import Settings


def test_settings_are_isolated_from_the_real_environment() -> None:
    settings = Settings()
    assert settings.telegram_bot_token is None
    assert settings.api_token is None
    assert settings.llm_api_key is None
    assert settings.slack_bot_token is None
    assert settings.slack_signing_secret is None
    assert settings.email_password is None
    assert settings.telegram_allowed_chat_ids == []
    assert settings.slack_allowed_user_ids == []


def test_explicit_values_still_apply() -> None:
    # Isolation must not break a test's ability to configure Settings directly.
    settings = Settings(telegram_bot_token="tok", api_token="sekrit")
    assert settings.telegram_bot_token == "tok"
    assert settings.api_token == "sekrit"


def test_defaults_keep_the_conservative_posture() -> None:
    settings = Settings()
    assert settings.llm_provider == "codex"
    assert settings.enable_email is False  # the only external-service subsystem
    assert settings.enable_email_send is False  # sending needs a second switch
