"""Runtime configuration, loaded from the environment / a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # --- Memory (the "brain") ---
    # Root directory for long-term memory notes + the SQLite index. Relative
    # paths resolve against the server's working directory.
    memory_dir: str = "memory"
    # Local, offline embedding model (fastembed / ONNX — no API key). 384-dim.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # How many notes to recall and inject per turn.
    recall_top_k: int = 5
    # Cosine-similarity floor for a recalled note to be considered relevant.
    recall_min_similarity: float = 0.35
    # Master switch for long-term memory upkeep. When True, an LLM extraction
    # runs after each turn (in the background) to save/forget notes — covering
    # both explicit "remember/forget …" and proactively captured facts.
    enable_auto_memory: bool = True

    # --- HTTP server ---
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def memory_path(self) -> Path:
        """Absolute path to the memory directory (created on first use)."""
        return Path(self.memory_dir).expanduser().resolve()

    @property
    def memory_db_path(self) -> Path:
        """SQLite file holding the vector index and the LangGraph checkpointer."""
        return self.memory_path / "index.db"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
