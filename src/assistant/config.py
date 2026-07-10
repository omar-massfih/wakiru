"""Runtime configuration, loaded from the environment / a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the assistant.

    Auth is handled by the Codex CLI itself (``codex login`` / ChatGPT sign-in),
    so there is no API key here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider selection ---
    # Which backend the agent's model uses. Wired: "codex", "openai", "anthropic".
    llm_provider: str = "codex"

    # --- API-backed providers (openai / anthropic) ---
    # Only used when llm_provider is "openai" or "anthropic"; the codex provider
    # authenticates through the Codex CLI itself and ignores these.
    # API key for the selected provider (OPENAI_API_KEY / ANTHROPIC_API_KEY also
    # work via the provider SDKs, but setting it here keeps config in one place).
    llm_api_key: str | None = None
    # Optional custom base URL — e.g. an OpenAI-compatible endpoint or a proxy.
    # Ignored by the anthropic provider.
    llm_base_url: str | None = None
    # Model name for the selected provider. None => a sensible per-provider
    # default (see llm.py: gpt-4o for openai, claude-opus-4-8 for anthropic).
    llm_model: str | None = None

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
    # Max Codex subprocesses running at once (each can block a worker thread
    # for up to codex_timeout seconds); excess calls queue for a slot.
    codex_max_concurrency: int = 4
    # Enable Codex's native web_search tool (live internet search, no
    # per-call approval). Off by default — costs extra tokens/latency;
    # widen deliberately, matching codex_sandbox's conservative default.
    codex_web_search: bool = False

    # --- Storage backend ---
    # local keeps the original markdown + SQLite stores. postgres is intended
    # for Vercel Marketplace databases such as Neon with pgvector enabled.
    storage_backend: str = "local"
    database_url: str | None = None

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
    # Cap per kind on how many index entries are injected into the prompt each
    # turn (-1 = unlimited, 0 = omit the kind entirely). Bounds the per-turn
    # context as notes accumulate; MEMORY.md on disk is never trimmed. Episodic
    # defaults to 0: raw traces are recalled semantically when relevant, and a
    # listing of them is noise.
    context_index_max_per_kind: dict[str, int] = {
        "semantic": 20,
        "procedural": 10,
        "episodic": 0,
    }

    # --- Dedup / forget thresholds (cosine; model-dependent!) ---
    # A new save whose nearest same-kind note scores >= this is treated as a
    # restatement and updates in place. Calibrated for e5-large, whose sentence
    # similarities cluster high (restatements ~0.97, distinct facts ~0.85). Lower
    # this for models with a wider similarity spread (e.g. ~0.85 for MiniLM).
    dedup_threshold: float = 0.90
    # How many nearest notes a save is deduped against.
    dedup_candidates: int = 10
    # A durable save matching a durable note of the OTHER kind at/above this
    # updates it in place (semantic <-> procedural only; episodic never merges).
    # Deliberately above dedup_threshold: crossing kinds needs near-certainty.
    dedup_cross_kind_threshold: float = 0.95
    # Floor for deleting a note by a fuzzy (non-exact-name) forget query. With
    # e5-large, distinct facts sit ~0.85 and restatements ~0.97, so 0.88 clears
    # unrelated facts while still catching paraphrased targets.
    forget_threshold: float = 0.88
    # Fuzzy forget is skipped (no-op) when the two best matches are this close —
    # deleting nothing beats deleting the wrong memory.
    forget_ambiguity_margin: float = 0.03

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
    # Skip the per-turn episodic trace for very short user messages (0 disables).
    episodic_min_chars: int = 12
    # Skip the trace when it near-duplicates an existing episode (cosine).
    episodic_dedup_threshold: float = 0.97
    # Hard per-kind note caps enforced during consolidation (0 = uncapped).
    # Lowest retention-score notes are evicted first.
    max_notes_per_kind: dict[str, int] = {
        "semantic": 200,
        "procedural": 100,
        "episodic": 200,
    }
    # Half-life (days) for the effective-salience decay of durable notes that
    # have never been recalled (0 disables). Decay never deletes a note.
    durable_decay_half_life_days: float = 180.0
    # Deleted notes are moved to memory/.trash (recoverable by hand) rather than
    # unlinked; consolidation permanently prunes them after this many days.
    trash_retention_days: int = 30

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

    # --- Email (opt-in; the only subsystem that talks to an external service) ---
    # Master switch. OFF by default: email is the one capability that needs real
    # external auth (OAuth2 / an app password) and reads private correspondence.
    # Nothing below is touched — and no connection is ever opened — while this is False.
    enable_email: bool = False
    # The mailbox address (also the IMAP/SMTP username).
    email_address: str | None = None
    # How to authenticate: "oauth" (XOAUTH2 via a refresh token) or "password"
    # (an app password / basic LOGIN).
    email_auth: str = "oauth"
    # IMAP (read + save-draft) and SMTP (send) endpoints. Defaults are Gmail's.
    email_imap_host: str = "imap.gmail.com"
    email_imap_port: int = 993
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    # The IMAP mailbox drafts are appended to (Gmail: "[Gmail]/Drafts").
    email_drafts_folder: str = "[Gmail]/Drafts"
    # OAuth2: a long-lived refresh token is exchanged for short-lived access
    # tokens, cached under the memory dir. Obtain these once, out of band.
    email_oauth_client_id: str | None = None
    email_oauth_client_secret: str | None = None
    email_oauth_refresh_token: str | None = None
    email_oauth_token_url: str = "https://oauth2.googleapis.com/token"
    # App password, when email_auth == "password".
    email_password: str | None = None
    # Cap on how many messages a listing returns.
    email_max_messages: int = 10
    # Second, independent gate for actually SENDING mail. Drafting is always
    # allowed when email is on; sending is not, unless this is deliberately set.
    # The assistant never sends in the background — only on an explicit request.
    enable_email_send: bool = False

    # --- Documents / notes ---
    # Master switch: ingest documents (chunked + embedded) and surface the most
    # relevant chunks into the recall context each turn ("what did I write about X").
    enable_docs: bool = True
    # How many document chunks to inject per turn.
    docs_recall_top_k: int = 3
    # Cosine-similarity floor for a chunk to be injected at all.
    docs_min_similarity: float = 0.35
    # Target size (characters) for a document chunk when ingesting. Must be
    # positive: a zero target makes the chunker's hard-split loop never terminate.
    docs_chunk_chars: int = Field(800, ge=1)
    # Chunk size (characters) for the map step when summarizing a long document.
    # Much larger than docs_chunk_chars — that target is tuned for retrieval, and
    # reusing it here would mean one model call per 800 characters.
    docs_summarize_chars: int = Field(8000, ge=1)

    # --- Tasks / to-dos ---
    # Master switch: inject open tasks into each turn (the read path) so the model
    # knows what's outstanding.
    enable_tasks: bool = True
    # Master switch for the write path: an LLM extraction after each turn that
    # adds/completes/updates/removes tasks (parallel to enable_auto_schedule).
    enable_auto_tasks: bool = True
    # Cap on how many open tasks are injected per turn.
    tasks_max_open: int = 20

    # --- Confirmation on writes (undo) ---
    # Master switch: log every calendar write to an undo ledger, let the user
    # revert the latest one by replying "undo", and push an out-of-band
    # confirmation (with the undo hint) after each background write.
    enable_write_confirmation: bool = True
    # How long after a write "undo" can still revert it (minutes).
    write_undo_window_minutes: int = 15

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

    # --- Telegram channel ---
    # Bot token from @BotFather. Set => the server long-polls Telegram and each
    # chat becomes a persistent conversation (thread "telegram:<chat_id>").
    # The first chat to message the bot is paired automatically (trust-on-first-
    # use, persisted under the memory dir); everyone else gets silence.
    telegram_bot_token: str | None = None
    # Optional explicit allowlist (JSON list, e.g. [123456789]), merged with the
    # paired set — use it to pin the owner up front or add extra chats. Proactive
    # reminders are pushed to all authorized chats.
    telegram_allowed_chat_ids: list[int] = []

    # --- Slack channel ---
    # Bot token (xoxb-…). Set => POST /slack/events accepts Slack Events API
    # callbacks and each user's DM/channel becomes a persistent conversation.
    slack_bot_token: str | None = None
    # Signing secret from the Slack app config; every callback's HMAC is verified
    # against it. Required whenever slack_bot_token is set.
    slack_signing_secret: str | None = None
    # Explicit allowlist of Slack user ids (e.g. ["U0123ABC"]) the bot answers.
    # Empty => the bot answers nobody: unlike Telegram there is no pairing
    # handshake, so an unconfigured allowlist must fail closed, not open.
    slack_allowed_user_ids: list[str] = []
    # Channel id reminders and write-confirmations are pushed to. None => no push.
    slack_notify_channel: str | None = None

    # --- HTTP server ---
    host: str = "127.0.0.1"
    port: int = 8000
    # Bearer token required on every endpoint except /health. None (the default)
    # keeps the legacy loopback-trust behavior — fine on 127.0.0.1, but set this
    # before binding to any non-loopback interface (the Docker CMD binds 0.0.0.0).
    api_token: str | None = None
    # Escape hatch: deliberately serve without a bearer token on a non-loopback
    # bind (e.g. behind a reverse proxy or VPN that does its own auth). Without
    # this, startup refuses that combination outright.
    allow_unauthenticated: bool = False

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
    def tasks_db_path(self) -> Path:
        """SQLite file holding the to-do list (tasks + their undo ledger)."""
        return self.memory_path / "tasks.db"

    @property
    def docs_db_path(self) -> Path:
        """SQLite file holding ingested documents + their chunk vector index."""
        return self.memory_path / "docs.db"

    @property
    def mail_token_path(self) -> Path:
        """Cache file for the short-lived OAuth2 access token (never the refresh token)."""
        return self.memory_path / "mail_token.json"

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
