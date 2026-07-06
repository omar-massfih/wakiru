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
    # Local, offline embedding model (fastembed / ONNX — no API key). The default
    # is multilingual (strong Norwegian recall), 1024-dim. It is an *asymmetric*
    # e5 model — embeddings.py adds the required query:/passage: prefixes.
    embedding_model: str = "intfloat/multilingual-e5-large"
    # How many notes to recall and inject per turn.
    recall_top_k: int = 5
    # Cosine-similarity floor for a candidate note to be considered at all.
    recall_min_similarity: float = 0.35
    # Master switch for long-term memory upkeep. When True, an LLM extraction
    # runs after each turn (in the background) to save/update/forget notes.
    enable_auto_memory: bool = True

    # --- Dedup / forget thresholds (cosine; model-dependent!) ---
    # A new save whose nearest same-kind note scores >= this is treated as a
    # restatement and updates in place. Calibrated for e5-large, whose sentence
    # similarities cluster high (restatements ~0.97, distinct facts ~0.85). Lower
    # this for models with a wider similarity spread (e.g. ~0.85 for MiniLM).
    dedup_threshold: float = 0.90
    # Floor for deleting a note by a fuzzy (non-exact-name) forget query.
    forget_threshold: float = 0.80

    # --- Retrieval ranking (blended re-rank on top of cosine) ---
    # Candidate pool size = recall_top_k * this, re-ranked then trimmed to top_k.
    recall_candidate_multiplier: int = 3
    recall_w_similarity: float = 1.0
    recall_w_recency: float = 0.15
    recall_w_reuse: float = 0.15
    recall_w_salience: float = 0.1
    # Half-life (days) for the recency signal.
    recall_recency_half_life_days: float = 30.0
    # Saturation point for the reuse (recall_count) signal.
    recall_reuse_cap: int = 20
    # Small additive bias per kind (favor durable memory over raw episodes).
    recall_kind_bias: dict[str, float] = {
        "semantic": 0.05,
        "procedural": 0.05,
        "episodic": -0.05,
    }

    # --- Consolidation ("sleep") ---
    # Run a consolidation pass every N chat turns (0 disables the auto-trigger).
    consolidate_every_n_turns: int = 20
    # Max episodic traces fed to the consolidation LLM in one pass.
    consolidate_max_episodes: int = 40
    # Episodic salience at creation, and the pruning floor / age horizon.
    episodic_initial_salience: float = 0.25
    salience_prune_floor: float = 0.05
    episodic_max_age_days: int = 30

    # --- Working memory (conversation history) ---
    # Summarize + trim history once it exceeds this many messages (0 disables).
    working_memory_max_messages: int = 40
    # How many of the most recent messages to keep verbatim after summarizing.
    working_memory_keep_recent: int = 12

    # --- Time & calendar ---
    # IANA timezone name (e.g. "Europe/Oslo") the assistant reasons in. None =>
    # the server's system-local timezone. Used for the current-time context it is
    # given each turn and for resolving natural-language dates when scheduling.
    timezone: str | None = None
    # Master switch: inject current time + upcoming events into each turn.
    enable_calendar: bool = True
    # Master switch for the write path: an LLM extraction after each turn that
    # creates/reschedules/cancels events (parallel to enable_auto_memory).
    enable_auto_schedule: bool = True
    # How far ahead (days) upcoming events are surfaced to the model.
    calendar_upcoming_days: int = 14
    # Cap on how many upcoming events are injected per turn.
    calendar_max_events: int = 20

    # --- Proactive reminders ---
    # Master switch: fire proactive reminders ahead of upcoming events.
    enable_reminders: bool = True
    # Fire a reminder this many minutes before an event. A list, so several leads
    # per event work (e.g. [1440, 60] = a day before and an hour before). The
    # default single lead means at most one push per event.
    reminder_lead_minutes: list[int] = [60]
    # ntfy topic URL / generic webhook the reminder is POSTed to. None => reminders
    # are still computed and returned by the endpoint, just not pushed anywhere.
    reminder_webhook_url: str | None = None
    # How often the in-process ticker fires run_reminders (seconds). 0 disables the
    # built-in ticker; POST /reminders/run still works for manual/external triggering.
    reminder_tick_seconds: int = 60

    # --- HTTP server ---
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def memory_path(self) -> Path:
        """Absolute path to the memory directory (created on first use)."""
        return Path(self.memory_dir).expanduser().resolve()

    @property
    def memory_db_path(self) -> Path:
        """SQLite file holding the vector index + reinforcement counters."""
        return self.memory_path / "index.db"

    @property
    def calendar_db_path(self) -> Path:
        """SQLite file holding the local calendar's events."""
        return self.memory_path / "calendar.db"

    @property
    def checkpoints_db_path(self) -> Path:
        """SQLite file holding the LangGraph conversation checkpoints.

        Kept separate from the vector index so background index writes never
        contend with the long-lived checkpointer connection, and a reindex can
        rebuild vectors without touching conversation history.
        """
        return self.memory_path / "checkpoints.db"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
