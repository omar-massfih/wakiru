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

ENV CODEX_HOME=/root/.codex
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
