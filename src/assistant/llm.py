"""LLM provider abstraction.

Every provider returns a LangChain ``BaseChatModel``, so the rest of the app (the
graph, the API) is provider-agnostic. ``build_model`` selects one via
``settings.llm_provider``.

Wired today:
  - ``codex``     — drives the Codex CLI (auth via ``codex login``; no API key).
  - ``openai``    — hosted OpenAI / any OpenAI-compatible endpoint via ChatOpenAI.
  - ``anthropic`` — Claude via ChatAnthropic.

The API-backed providers read their key/model/base-url from ``Settings``
(``llm_api_key`` / ``llm_model`` / ``llm_base_url``); the codex provider ignores
those and authenticates through the Codex CLI itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from collections.abc import Iterator

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from .codex_runner import run_codex, run_codex_stream
from .config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Codex provider
# --------------------------------------------------------------------------- #

_ROLE_LABELS = {
    "system": "System",
    "human": "User",
    "ai": "Assistant",
    "tool": "Tool",
}


def _render_prompt(messages: list[BaseMessage]) -> str:
    """Flatten a message list into a single prompt string for ``codex exec``."""
    lines: list[str] = []
    for message in messages:
        label = _ROLE_LABELS.get(message.type, message.type.capitalize())
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"{label}: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


class CodexChatModel(BaseChatModel):
    """A ``BaseChatModel`` that delegates generation to the Codex CLI."""

    settings: Settings

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _render_prompt(messages)
        text = run_codex(prompt, settings=self.settings)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # The subprocess call blocks; without this override, ainvoke would run
        # it on the event loop's default executor with no clear ownership.
        return await asyncio.to_thread(self._generate, messages, stop, None, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Having this override is what makes invoke() stream: BaseChatModel's
        # _should_stream routes through here whenever a streaming callback
        # handler (e.g. LangGraph's messages mode) is attached. The async side
        # needs no override — astream falls back to running this in an executor.
        prompt = _render_prompt(messages)
        for delta in run_codex_stream(prompt, settings=self.settings):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk


def _build_codex(settings: Settings) -> BaseChatModel:
    return CodexChatModel(settings=settings)


# --------------------------------------------------------------------------- #
# API-backed providers
# --------------------------------------------------------------------------- #

# Per-provider default model when settings.llm_model is unset.
_DEFAULT_OPENAI_MODEL = "gpt-4o"
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


def _require_api_key(settings: Settings, provider: str) -> str:
    key = settings.llm_api_key
    if not key:
        raise ValueError(
            f"LLM_PROVIDER={provider!r} requires LLM_API_KEY to be set "
            "(the API key for the provider)."
        )
    return key


def _build_openai(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model or _DEFAULT_OPENAI_MODEL,
        api_key=_require_api_key(settings, "openai"),
        # None => the SDK's default endpoint; set for an OpenAI-compatible proxy.
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=settings.codex_timeout,
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # No temperature: the current Claude models (Opus 4.8 etc.) reject non-default
    # sampling params, and ChatAnthropic omits it unless set.
    return ChatAnthropic(
        model=settings.llm_model or _DEFAULT_ANTHROPIC_MODEL,
        api_key=_require_api_key(settings, "anthropic"),
        max_tokens=4096,
        timeout=settings.codex_timeout,
    )


# --------------------------------------------------------------------------- #
# Registry + factory
# --------------------------------------------------------------------------- #

ProviderBuilder = Callable[[Settings], BaseChatModel]

PROVIDERS: dict[str, ProviderBuilder] = {
    "codex": _build_codex,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
}


def build_model(settings: Settings | None = None) -> BaseChatModel:
    """Construct the chat model for the configured provider."""
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    builder = PROVIDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}. "
            f"Options: {', '.join(sorted(PROVIDERS))}."
        )
    return builder(settings)
