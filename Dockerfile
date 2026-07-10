# Agentic assistant — LangGraph/LangChain over the Codex CLI.
#
# The image bundles the Codex CLI, but NOT its credentials. Codex authenticates via
# `codex login` (ChatGPT), which stores tokens in CODEX_HOME (~/.codex). Mount your
# host credentials at runtime — see the run command at the bottom.

FROM python:3.13-slim

# System deps:
#   - nodejs/npm : the Codex CLI is distributed as an npm package
#   - git        : some Codex flows expect a git context
#   - ca-certificates/curl : TLS + fetching uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Codex CLI
RUN npm install -g @openai/codex && codex --version

# uv (pinned via the published image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer (cached unless lockfile changes). Prod deps only — skip dev group.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application
COPY . .
RUN uv sync --frozen --no-dev

# Run as a non-root user. Codex credentials are mounted at its ~/.codex, and the
# memory store lives under /app/memory (created here so the volume is writable).
RUN useradd --create-home --uid 1000 assistant \
    && mkdir -p /app/memory \
    && chown -R assistant:assistant /app
USER assistant
ENV CODEX_HOME=/home/assistant/.codex
# api.lifespan reads HOST to gate non-loopback serving on API_TOKEN.
ENV HOST=0.0.0.0
EXPOSE 8000

# --no-dev: match the build, or `uv run` re-syncs mypy/ruff/pytest on every start.
CMD ["uv", "run", "--no-dev", "uvicorn", "assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
