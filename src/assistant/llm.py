"""LLM provider abstraction.

Every provider returns a LangChain ``BaseChatModel``, so the rest of the app (the
graph, the API) is provider-agnostic. ``build_model`` selects one via
``settings.llm_provider``.

Wired today:
  - ``codex``  — drives the Codex CLI (auth via ``codex login``; no API key).

Ready to add (stubs below say exactly what to fill in):
  - ``openai``    — hosted OpenAI / any OpenAI-compatible endpoint via ChatOpenAI.
  - ``anthropic`` — Claude via ChatAnthropic.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .codex_runner import run_codex
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


def _build_codex(settings: Settings) -> BaseChatModel:
    return CodexChatModel(settings=settings)


# --------------------------------------------------------------------------- #
# Future API-backed providers (stubs)
# --------------------------------------------------------------------------- #


def _build_openai(settings: Settings) -> BaseChatModel:
    # To enable:
    #   1) uv add langchain-openai
    #   2) add settings fields (e.g. llm_api_key, llm_base_url, llm_model)
    #   3) replace the body below with:
    #        from langchain_openai import ChatOpenAI
    #        return ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key,
    #                          base_url=settings.llm_base_url, temperature=0)
    raise NotImplementedError(
        "openai provider is not wired yet — see the instructions in llm.py:_build_openai."
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    # To enable:
    #   1) uv add langchain-anthropic
    #   2) add settings fields (e.g. llm_api_key, llm_model)
    #   3) replace the body below with:
    #        from langchain_anthropic import ChatAnthropic
    #        return ChatAnthropic(model=settings.llm_model, api_key=settings.llm_api_key,
    #                             max_tokens=4096)
    raise NotImplementedError(
        "anthropic provider is not wired yet — see the instructions in llm.py:_build_anthropic."
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
