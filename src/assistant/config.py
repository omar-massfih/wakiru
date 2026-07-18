"""Runtime configuration, loaded from the environment / a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the assistant.

    For the codex and chatgpt providers, auth is handled by the Codex CLI's
    login (``codex login`` / ChatGPT sign-in), so there is no API key here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider selection ---
    # Which backend the agent's model uses.
    # Wired: "codex", "chatgpt", "openai", "anthropic".
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
    # Request timeout (seconds) for the API-backed providers. The codex
    # provider has its own wall-clock cap, codex_timeout below.
    llm_timeout: int = 120
    # Reply-length cap (tokens) for the API-backed providers.
    llm_max_tokens: int = 4096

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

    # --- ChatGPT backend (llm_provider="chatgpt") ---
    # Talks to chatgpt.com's backend directly, reusing the OAuth tokens the
    # Codex CLI keeps in $CODEX_HOME/auth.json (run `codex login` once).
    # Usage and rate limits come from the ChatGPT plan, not API billing.
    # The backend only accepts codex-supported model ids ("gpt-5.5" as of
    # codex-cli 0.144.5); anything else gets an HTTP 400.
    chatgpt_model: str = "gpt-5.5"
    # Override the auth file location. None => $CODEX_HOME/auth.json,
    # falling back to ~/.codex/auth.json.
    chatgpt_auth_file: str | None = None
    # Socket timeout (seconds) for a single request/stream read.
    chatgpt_timeout: int = 300
    # Max concurrent requests; excess calls queue for a slot (bursts would
    # otherwise eat into the ChatGPT plan's rate limits all at once).
    chatgpt_max_concurrency: int = 4

    # --- Persona ---
    # Voice register for the assistant's replies: "warm" (default), "neutral",
    # or "minimal". Unknown values fall back to warm. The system prompt stays
    # byte-stable per value, so prompt caching keeps paying off.
    persona_style: str = "warm"

    # --- Tool loop ---
    # Max tool rounds per turn; past it, pending calls get a budget-exhausted
    # result and the model must answer with what it has.
    tool_max_rounds: int = 6

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
    # Recall query expansion: how many recent turns (besides the latest message)
    # contribute snippets to the embedding query, and the total budget (chars)
    # for that supplement (0 disables — the latest message alone is the query).
    recall_context_messages: int = 4
    recall_context_extra_chars: int = 600
    # Cosine-similarity floor for a candidate note to be considered at all.
    recall_min_similarity: float = 0.35
    # Master switch for long-term memory upkeep. When True, an LLM extraction
    # runs after each turn (in the background) to save/update/forget notes.
    enable_auto_memory: bool = True
    # Personalization: inject durable notes tagged "profile" (working hours,
    # locations, quiet hours, tone) into every turn, and let the reminder/briefing
    # tickers honor stated quiet hours. Degrades to a no-op with no such notes.
    enable_profile: bool = True
    # Default quiet window (local time, "HH:MM-HH:MM") used when no profile note
    # states one. A stated preference in memory always wins. Empty string
    # disables the default (legacy behavior: no note => never quiet).
    quiet_hours_default: str = "22:00-07:30"
    # Cap per kind on how many index entries are injected into the prompt each
    # turn (-1 = unlimited, 0 = omit the kind entirely). Bounds the per-turn
    # context as notes accumulate; MEMORY.md on disk is never trimmed. Episodic
    # is kept small: raw traces are recalled semantically when relevant, but a
    # couple of high-retention episodes in the listing let the model know what
    # recent sessions it can draw on.
    context_index_max_per_kind: dict[str, int] = {
        "semantic": 20,
        "procedural": 10,
        "episodic": 2,
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
    # Clock-driven sleep: a nightly maintenance pass (working-memory folding +
    # consolidation) that runs without a user turn, so a quiet week still gets
    # memory upkeep instead of none. Rides its own slow loop, exactly-once per
    # local date via a fired ledger, and — unlike reminders/briefing — runs
    # *during* quiet hours by design (it never pushes anything).
    enable_sleep: bool = True
    # Local wall-clock time (HH:MM, in TIMEZONE) the nightly pass becomes due.
    # Defaults inside the default quiet window on purpose.
    sleep_time: str = "03:30"
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
    # How far ahead (days) upcoming events are surfaced to the model.
    calendar_upcoming_days: int = 14
    # Cap on how many upcoming events are injected per turn.
    calendar_max_events: int = 20
    # ICS subscription URLs (Google's "secret iCal address", Outlook, any CalDAV
    # export) mirrored into the local calendar, read-only, on the sync cadence
    # below. Empty => no external sync.
    calendar_ics_urls: list[str] = []
    # Minutes between feed pulls (the sync rides the reminder ticker). 0 disables
    # the automatic pull; POST /calendar/sync still works.
    calendar_sync_minutes: int = 15

    # --- Two-way CalDAV sync (opt-in; second switch for writes) ---
    # Master switch for the CalDAV round-trip: pull one writable collection into the
    # local store as read+write rows (distinct from the read-only ICS mirror above).
    # OFF by default — CalDAV needs real external auth, like email.
    enable_caldav: bool = False
    # SECOND, independent gate: push local create/reschedule/cancel back to the
    # collection. Reads work with just enable_caldav; nothing is ever written to the
    # remote unless this is deliberately set. The reply path pushes; the background
    # loop only reconciles already user-intended writes.
    enable_caldav_write: bool = False
    # Which remote-calendar transport to use: "caldav" (RFC 4791, for Fastmail/
    # iCloud/Nextcloud) or "google" (the Google Calendar REST API v3, because Google
    # walls off CalDAV — v1 is dead and v2 returns 403 to normal OAuth apps). The
    # "google" provider reuses the OAuth credentials below.
    caldav_provider: str = "caldav"
    # Google provider only: which calendar to sync ("primary", or a calendar id /
    # email). Ignored for the CalDAV provider.
    google_calendar_id: str = "primary"
    # The writable calendar *collection* URL (not a principal/home-set). https, and
    # validated by the SSRF guard like every other outbound fetch. CalDAV provider only.
    caldav_url: str | None = None
    # Basic-auth credentials (an app-specific password on Fastmail/iCloud/Nextcloud).
    caldav_username: str | None = None
    caldav_password: str | None = None
    # How to authenticate: "password" (Basic, for Fastmail/iCloud/Nextcloud) or
    # "oauth" (Bearer, for Google — which does not allow password CalDAV).
    caldav_auth: str = "password"
    # Minutes between CalDAV pulls + outbox reconcile. 0 disables the loop
    # (POST /calendar/sync still runs it).
    caldav_sync_minutes: int = 15
    # OAuth2 (Google CalDAV): a long-lived refresh token is exchanged for short-lived
    # access tokens, cached under the memory dir — the same flow as mail/oauth.py. Used
    # only when caldav_auth == "oauth". Obtain the refresh token once, out of band, with
    # scripts/caldav_oauth_setup.py. For Google, caldav_url is the primary calendar's
    # events collection: https://apps.google.com/calendar/dav/<you@gmail.com>/events/
    caldav_oauth_client_id: str | None = None
    caldav_oauth_client_secret: str | None = None
    caldav_oauth_refresh_token: str | None = None
    caldav_oauth_token_url: str = "https://oauth2.googleapis.com/token"

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
    # Minutes between background refreshes of the unread-mail snapshot that is
    # injected into each turn's context (IMAP never runs on the reply path).
    # 0 disables both the refresh and the per-turn mail block.
    email_snapshot_minutes: int = 15
    # Where archive_email moves mail on non-Gmail servers. Gmail archives by
    # removing the \Inbox label instead (the message stays in All Mail), so
    # this folder is never used there.
    email_archive_folder: str = "Archive"
    # IMAP dialect: "auto" sniffs Gmail's X-GM-EXT-1 capability on the
    # authenticated connection; "gmail" / "generic" force one path (useful
    # behind proxies that hide capabilities).
    email_provider: str = "auto"
    # Max autonomous mailbox mutations (archive/label/mark-read/draft-reply)
    # per background wake. 0 (default) keeps the background strictly read-only:
    # the mutating mail tools are absent from the heartbeat registry entirely.
    # Chat mode is unaffected. Sending is never possible in the background.
    email_triage_max_actions: int = 0

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
    # Allow POST /documents to ingest a URL (fetched server-side, HTML reduced to
    # prose). Off by default: on a server others can reach, fetching arbitrary
    # URLs on request is an SSRF primitive.
    enable_docs_url_ingest: bool = False
    # Byte cap for POST /documents/upload. Larger than the 2 MB text-body cap
    # because PDF/DOCX containers are bulkier than the prose they yield.
    docs_upload_max_bytes: int = Field(10_000_000, ge=1)

    # --- Tasks / to-dos ---
    # Master switch: inject open tasks into each turn (the read path) so the model
    # knows what's outstanding.
    enable_tasks: bool = True
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
    # per event work (e.g. [1440, 60] = a day before and an hour before). With
    # importance classification on (below) this is the NORMAL-tier schedule; with
    # it off, the uniform schedule for every event.
    reminder_lead_minutes: list[int] = [15]
    # Master switch: classify each event's importance with the LLM (one call per
    # new event, cached in calendar.db) and pick its lead schedule per tier.
    # False => every event uses reminder_lead_minutes (uniform legacy behavior).
    reminder_importance_enabled: bool = True
    # Lead schedule for critical events (doctor, flight, exam, deadline …):
    # 2 days, 1 day, 3 hours, 1 hour, 15 minutes before.
    reminder_lead_minutes_critical: list[int] = [2880, 1440, 180, 60, 15]
    # ntfy topic URL / generic webhook the reminder is POSTed to. None => reminders
    # are still computed and returned by the endpoint, just not pushed anywhere.
    reminder_webhook_url: str | None = None
    # How often the in-process ticker fires run_reminders (seconds). 0 disables the
    # built-in ticker; POST /reminders/run still works for manual/external triggering.
    reminder_tick_seconds: int = 60
    # When > 0, repeat reminders every N minutes once an item is inside its lead
    # window, instead of firing each lead once. reminder_lead_minutes then defines
    # only when reminders BEGIN (its max); the repeats fill the countdown until the
    # event starts / a task is done. 0 keeps the legacy one-shot-per-lead path.
    reminder_repeat_minutes: int = 0
    # How long past its due time an open task keeps nagging (minutes). Bounds the
    # overdue repeat so a forgotten task can't nudge forever. Only used when
    # reminder_repeat_minutes > 0.
    reminder_overdue_max_minutes: int = 1440  # 24h

    # --- Daily briefing ---
    # Push one proactive digest per day (agenda + due tasks + unread mail when
    # email is on) through the same channels reminders use.
    enable_briefing: bool = False
    # Local wall-clock time (HH:MM, in TIMEZONE) the briefing becomes due. The
    # ticker fires it on the first tick at/after this time, exactly once per
    # day. With the heartbeat enabled the model composes the briefing in its
    # own voice; without it the assembled digest goes out verbatim (no LLM).
    briefing_time: str = "07:30"
    # Record proactive pushes (reminders, the briefing) into each authorized
    # chat's working memory, so the conversation knows what was already sent
    # ("what was that reminder about?" works). No extra LLM cost.
    enable_proactive_loop_in: bool = True

    # --- Heartbeat (deliberative proactivity) ---
    # Periodically let the model itself review the situation (due followups,
    # briefing, mail changes, contact staleness — or nothing at all) and
    # decide whether reaching out helps — or stay silent. Reminders keep
    # running regardless: they are the reflex arc, this is the deliberative
    # layer. Off by default; also unlocks the schedule_followup tools.
    enable_heartbeat: bool = False
    # Minutes between heartbeat wakes. EVERY wake is a model call (only quiet
    # hours and an all-scope mute hold it), so this is the direct token-cost
    # dial. 0 disables the ticker (POST /heartbeat/run still works).
    heartbeat_minutes: int = 30
    # Minimum minutes between ambient heartbeat *pushes* (a message the model
    # composed with no due follow-up or briefing behind it), so a chatty model
    # doesn't become a barrage. Bounds delivery, never the model's judgment;
    # due follow-ups and the briefing always deliver regardless of the gap.
    heartbeat_min_gap_minutes: int = 120
    # Hours of user silence (across all channels) before "we haven't talked in
    # a while" becomes a heartbeat trigger. 0 disables the staleness trigger.
    heartbeat_contact_gap_hours: int = 0
    # Self-pacing: the model can set its own next wake (the set_next_wake tool),
    # and the scheduler never sleeps past the soonest open follow-up. These clamp
    # how far a self-set wake may move from HEARTBEAT_MINUTES. The floor keeps a
    # self-set wake from busy-looping the model; the ceiling caps how long it may
    # back off — 0 means "no later than the fixed cadence" (the model may only
    # pull the next wake *earlier*), so raising it (e.g. 360) is what lets the
    # model save tokens by backing off on quiet days.
    heartbeat_wake_min_minutes: int = 5
    heartbeat_wake_max_minutes: int = 0

    # --- Goals (standing multi-step intentions) ---
    # With the heartbeat on, the model can carry ongoing projects ("plan the
    # trip") in a goals store and advance them across wakes: the open_goal /
    # update_goal / close_goal tools, a per-turn context block, and a heartbeat
    # trigger when a goal's next_action_at comes due. The cap bounds how many
    # projects it may carry at once — every open goal rides along in every
    # prompt, so this is a token dial as much as a focus dial.
    goals_max_open: int = 5
    # Days without a goal update before the heartbeat starts nudging the model
    # to advance, reschedule, or abandon it (never auto-closed). 0 disables.
    goal_stale_days: int = 7

    # --- Watches (model-registered perception) ---
    # With the heartbeat on, the model can register its own observations
    # (watch/unwatch tools): "tell me when this sender writes", "wake me
    # before that event", "flag it if the user stays silent past a deadline".
    # Evaluation is deterministic and token-free on every wake; these bound
    # how many watches may be active at once and how long one may live
    # without an explicit expiry.
    watches_max_active: int = 10
    watch_default_expiry_days: int = 14

    # --- Reflection (nightly outcome review) ---
    # Once per night (riding the sleep pass) review how the day's proactive
    # behavior landed — pushes answered or ignored, writes undone, mutes with
    # their stated reasons — and distill at most a few "self" procedural notes
    # the model reads back on every wake. One LLM call per night at most; a day
    # with no proactive activity skips the call entirely.
    enable_reflection: bool = True
    # Cap on memory operations one reflection pass may apply.
    reflection_max_ops: int = 3

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
    # Merge consecutive plain-text messages from the same chat (typically sent
    # while a reply was still being composed) into one turn, so a thought split
    # across several messages gets one considered answer instead of fragments.
    telegram_coalesce_messages: bool = True

    # --- Voice notes (Telegram) ---
    # Transcribe incoming Telegram voice messages locally (faster-whisper) and
    # answer them like typed text. Off by default: first use downloads the model.
    enable_voice: bool = False
    # Whisper model size: "small" is the sweet spot for Norwegian + English;
    # "base"/"tiny" are lighter but noticeably worse outside English.
    voice_model: str = "small"
    # Force a language code ("no", "en", …). Empty => autodetect per message.
    voice_language: str = ""
    # Refuse clips longer than this (seconds): transcription is CPU-bound and a
    # very long clip would stall the reply loop.
    voice_max_seconds: int = 600

    # --- Slack channel ---
    # Bot token (xoxb-…). Set => POST /slack/events accepts Slack Events API
    # callbacks and each user's DM/channel becomes a persistent conversation.
    slack_bot_token: str | None = None
    # Signing secret from the Slack app config; every callback's HMAC is verified
    # against it. Required whenever slack_bot_token is set.
    slack_signing_secret: str | None = None
    # App-level token (xapp-…) with connections:write. Set (with slack_bot_token)
    # => events arrive over a Socket Mode websocket instead of requiring a public
    # HTTPS URL for POST /slack/events — works behind NAT, like Telegram polling.
    slack_app_token: str | None = None
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
    # Escape hatch: without this, startup refuses a non-loopback bind with no
    # token. Set it only when something in front (proxy, VPN) does the auth.
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
    def briefing_db_path(self) -> Path:
        """SQLite file holding the daily fired ledgers (briefing, nightly sleep)."""
        return self.memory_path / "briefing.db"

    @property
    def mutes_db_path(self) -> Path:
        """SQLite file holding active reminder mutes (see assistant.mutes)."""
        return self.memory_path / "mutes.db"

    @property
    def threads_db_path(self) -> Path:
        """SQLite file holding the live-thread registry (see assistant.threads)."""
        return self.memory_path / "threads.db"

    @property
    def followups_db_path(self) -> Path:
        """SQLite file holding the assistant's followups (see assistant.followups)."""
        return self.memory_path / "followups.db"

    @property
    def mail_db_path(self) -> Path:
        """SQLite file holding the mailbox-mutation audit ledger (see assistant.mail.audit)."""
        return self.memory_path / "mail.db"

    @property
    def mail_token_path(self) -> Path:
        """Cache file for the short-lived OAuth2 access token (never the refresh token)."""
        return self.memory_path / "mail_token.json"

    @property
    def caldav_token_path(self) -> Path:
        """Cache file for the short-lived CalDAV OAuth2 access token (0600, like mail)."""
        return self.memory_path / "caldav_token.json"

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


def postgres_backend(settings: Settings):
    """Return the ``storage_postgres`` module when ``STORAGE_BACKEND=postgres``, else ``None``.

    The local sqlite stores dispatch to their Postgres twins through this: a
    ``if pg := postgres_backend(settings):`` guard replaces the copy-pasted
    ``storage_backend == "postgres"`` check plus lazy import. The import stays
    lazy so the sqlite path never pulls in psycopg, and the *module* is returned
    (not pre-bound functions) so tests that monkeypatch ``storage_postgres``
    attributes are still resolved at call time.
    """
    if settings.storage_backend != "postgres":
        return None
    from . import storage_postgres

    return storage_postgres
