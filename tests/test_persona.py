"""Persona tests — the single identity/capability prompt and its flag gating."""

from __future__ import annotations

from assistant import persona
from assistant.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(memory_dir="memory", **overrides)


def test_prompt_is_byte_stable_per_configuration() -> None:
    settings = _settings()
    assert persona.system_prompt(settings) == persona.system_prompt(settings)


def test_identity_and_memory_always_present() -> None:
    prompt = persona.system_prompt(
        _settings(enable_calendar=False, enable_tasks=False, enable_docs=False)
    )
    assert "You are Wakiru" in prompt
    assert "How your memory works" in prompt
    assert "Acting with tools" in prompt
    assert "Initiative:" in prompt


def test_capability_sections_follow_their_flags() -> None:
    on = persona.system_prompt(_settings())
    assert "Calendar:" in on and "Tasks:" in on and "Documents:" in on

    off = persona.system_prompt(
        _settings(enable_calendar=False, enable_tasks=False, enable_docs=False)
    )
    assert "Calendar:" not in off and "Tasks:" not in off and "Documents:" not in off
    assert "create_event" not in off and "add_task" not in off


def test_email_section_and_send_gate() -> None:
    assert "Email:" not in persona.system_prompt(_settings())  # off by default

    draft_only = persona.system_prompt(_settings(enable_email=True))
    assert "Email:" in draft_only and "send_email" not in draft_only

    sending = persona.system_prompt(
        _settings(enable_email=True, enable_email_send=True)
    )
    assert "send_email" in sending and "never send unprompted" in sending


def test_reminder_etiquette_follows_reminders_flag() -> None:
    assert "⏰" in persona.system_prompt(_settings())
    assert "⏰" not in persona.system_prompt(_settings(enable_reminders=False))
    # No calendar and no tasks means nothing ever nudges.
    assert "⏰" not in persona.system_prompt(
        _settings(enable_calendar=False, enable_tasks=False)
    )


def test_undo_hint_follows_write_confirmation() -> None:
    on = persona.system_prompt(_settings(write_undo_window_minutes=15))
    assert 'reply "undo" within 15 minutes' in on
    off = persona.system_prompt(_settings(enable_write_confirmation=False))
    assert "undo" not in off.lower().replace("undone", "")


def test_system_message_is_cache_marked_on_anthropic() -> None:
    plain = persona.system_message(_settings())
    assert isinstance(plain.content, str)

    marked = persona.system_message(
        _settings(llm_provider="anthropic", llm_api_key="sk-test")
    )
    assert marked.content[0]["cache_control"] == {"type": "ephemeral"}
    assert marked.content[0]["text"].startswith("You are Wakiru")
