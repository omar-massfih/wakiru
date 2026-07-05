"""Runtime configuration, loaded from the environment / a local .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the assistant.

    Auth is handled by the Codex CLI itself (``codex login`` / ChatGPT sign-in),
    so there is no API key here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider selection ---
    # Which backend the agent's model uses. Wired: "codex".
    # Future: "openai", "anthropic" (see llm.py stubs).
    llm_provider: str = "codex"

    # --- Codex CLI ---
    codex_bin: str = "codex"
    # None => let Codex use whatever model its own config selects.
    codex_model: str | None = None
    # read-only is the safe default for a chat assistant; widen to
    # "workspace-write" / "danger-full-access" deliberately.
    codex_sandbox: str = "read-only"
    # Working root for Codex. None => the server's current working directory.
    codex_working_dir: str | None = None
    # Hard wall-clock cap (seconds) on a single Codex invocation.
    codex_timeout: int = 300

    # --- HTTP server ---
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
