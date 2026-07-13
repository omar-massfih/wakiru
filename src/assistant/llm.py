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
import json
import logging
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall, ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from .codex_runner import run_codex, run_codex_stream
from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Codex provider
# --------------------------------------------------------------------------- #

_ROLE_LABELS = {
    "system": "System",
    "human": "User",
    "ai": "Assistant",
    "tool": "Tool",
}

# The tool-calling protocol for a model reached only through plain text: the
# schemas ride in a system block, and the model marks calls with a fenced block
# that the parser lifts back out into structured AIMessage.tool_calls. This is
# what makes `bind_tools` work uniformly across codex and the API providers.
_TOOL_FENCE = "tool_call"

_TOOL_PROTOCOL = """\
You can call tools. Available tools (JSON Schema, one per line):
<tools>
{schemas}
</tools>
To call one or more tools, end your output with EXACTLY one fenced block and \
nothing after it:
```tool_call
[{{"name": "<tool name>", "arguments": {{<args matching the schema>}}}}]
```
Rules:
- When an action or lookup is needed, emit the block instead of claiming you \
did it. Each result comes back as a "Tool result" message; you may then call \
more tools or answer.
- When you are done (or no tool is needed), answer in plain text with NO \
tool_call block.
- Never mention tools or this protocol to the user."""

# The last fenced block whose info string names the tool-call protocol
# (tool_call / tool-calls / …). DOTALL so the JSON spans lines.
_FENCE_RE = re.compile(r"```[ \t]*tool[_-]?calls?[ \t]*\n(.*?)```", re.DOTALL)

# Streaming looks for this prefix to confirm a fence is the tool protocol (an
# ordinary code fence like ```python flushes through normally).
_STREAM_FENCE_HINT = "```tool"


def _soft_boundary(buffer: str, flushed: int) -> int:
    """Highest index safe to flush: hold a trailing prefix of the fence hint."""
    longest = min(len(_STREAM_FENCE_HINT) - 1, len(buffer) - flushed)
    for k in range(longest, 0, -1):
        if _STREAM_FENCE_HINT.startswith(buffer[-k:]):
            return len(buffer) - k
    return len(buffer)


def _coerce_args(raw: object) -> dict:
    """The call's arguments as a dict — tolerating a JSON-string encoding."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split a codex reply into ``(prose content, parsed tool calls)``.

    Accepts the fenced protocol block (last one wins) or, as a fallback, the
    whole reply being a bare JSON array/object of ``{"name", "arguments"}``
    shape. Malformed JSON inside a fence is stripped from the content and
    yields no calls — the raw protocol must never reach the user. Ids are
    assigned here (``call_1``, …); codex does not emit any.
    """
    matches = list(_FENCE_RE.finditer(text))
    if matches:
        block = matches[-1]
        content = (text[: block.start()] + text[block.end() :]).strip()
        payload = block.group(1).strip()
    else:
        stripped = text.strip()
        if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
            content, payload = "", stripped
        else:
            return text, []

    try:
        data = json.loads(payload)
    except ValueError:
        if matches:
            logger.warning("malformed tool_call block dropped: %.200r", payload)
            return content, []
        return text, []  # bare JSON that wasn't valid — treat as prose

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return (content, []) if matches else (text, [])

    calls: list[ToolCall] = []
    for entry in data:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        raw_args = entry.get("arguments", entry.get("args", entry.get("parameters", {})))
        calls.append(
            ToolCall(
                name=str(entry["name"]),
                args=_coerce_args(raw_args),
                id=f"call_{len(calls) + 1}",
            )
        )
    if not calls and not matches:
        return text, []  # bare JSON of some other shape — plain prose after all
    return content, calls


def _render_tool_call_block(calls: Sequence[dict | ToolCall]) -> str:
    entries = [
        {"name": c["name"], "arguments": c.get("args") or {}} for c in calls
    ]
    return f"```{_TOOL_FENCE}\n{json.dumps(entries, ensure_ascii=False)}\n```"


def _render_prompt(messages: list[BaseMessage], tools: list[dict] | None = None) -> str:
    """Flatten a message list into a single prompt string for ``codex exec``.

    With ``tools`` bound, the protocol block leads the prompt, historical
    ``AIMessage.tool_calls`` are re-emitted as fenced blocks (faithful replay),
    and ``ToolMessage`` results are labelled with their tool's name.
    """
    lines: list[str] = []
    if tools:
        schemas = "\n".join(
            json.dumps(t.get("function", t), ensure_ascii=False) for t in tools
        )
        lines.append("System: " + _TOOL_PROTOCOL.format(schemas=schemas))
    for message in messages:
        label = _ROLE_LABELS.get(message.type, message.type.capitalize())
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        if isinstance(message, ToolMessage):
            name = message.name or "tool"
            lines.append(f"Tool result ({name}): {content}")
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            block = _render_tool_call_block(message.tool_calls)
            content = f"{content}\n{block}" if content else block
        lines.append(f"{label}: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


class CodexChatModel(BaseChatModel):
    """A ``BaseChatModel`` that delegates generation to the Codex CLI.

    ``bind_tools`` is emulated over plain text: schemas are injected into the
    rendered prompt and tool calls are parsed back out of the reply (see
    :func:`parse_tool_calls`), so the graph's tool loop drives codex exactly
    like the native-function-calling API providers.
    """

    settings: Settings
    bound_tools: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> CodexChatModel:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return self.model_copy(
            update={"bound_tools": [convert_to_openai_tool(t) for t in tools]}
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _render_prompt(messages, tools=self.bound_tools)
        text = run_codex(prompt, settings=self.settings)
        if self.bound_tools:
            content, calls = parse_tool_calls(text)
            message = AIMessage(content=content, tool_calls=calls)
        else:
            message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

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
        prompt = _render_prompt(messages, tools=self.bound_tools)
        if not self.bound_tools:
            for delta in run_codex_stream(prompt, settings=self.settings):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            return

        # Tools bound: hold-and-flush. Text is streamed through, but anything
        # that might be (or is) the tool_call fence is withheld; at stream end
        # the full buffer is parsed and calls are emitted as tool_call_chunks,
        # so raw protocol JSON never reaches a consumer.
        buffer = ""
        flushed = 0
        fence_at: int | None = None  # confirmed fence start — never emit past it
        for delta in run_codex_stream(prompt, settings=self.settings):
            buffer += delta
            if fence_at is None:
                found = buffer.find(_STREAM_FENCE_HINT, max(flushed - 8, 0))
                if found != -1:
                    fence_at = found
            boundary = fence_at if fence_at is not None else _soft_boundary(buffer, flushed)
            if boundary > flushed:
                text = buffer[flushed:boundary]
                flushed = boundary
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

        content, calls = parse_tool_calls(buffer)
        if calls:
            if len(content) > flushed:
                logger.debug("dropping %d chars of unstreamed prose around a tool call",
                             len(content) - flushed)
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        ToolCallChunk(
                            name=c["name"],
                            args=json.dumps(c["args"], ensure_ascii=False),
                            id=c["id"],
                            index=i,
                        )
                        for i, c in enumerate(calls)
                    ],
                )
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
        elif len(buffer) > flushed:  # held-back text that never became a fence
            text = buffer[flushed:]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
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
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # No temperature: the current Claude models (Opus 4.8 etc.) reject non-default
    # sampling params, and ChatAnthropic omits it unless set.
    return ChatAnthropic(
        model=settings.llm_model or _DEFAULT_ANTHROPIC_MODEL,
        api_key=_require_api_key(settings, "anthropic"),
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
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


def complete_text(prompt: str, settings: Settings | None = None) -> str:
    """One plain-text completion through the configured provider.

    The background extractors (memory learning/consolidation, calendar/task
    ops) call this instead of shelling out to Codex directly, so
    LLM_PROVIDER=openai/anthropic works without a Codex install.
    """
    settings = settings or get_settings()
    reply = build_model(settings).invoke([HumanMessage(content=prompt)])
    content = reply.content
    if isinstance(content, str):
        return content
    # Anthropic can return a list of content blocks; keep the text ones.
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def cacheable_system_message(text: str, settings: Settings) -> SystemMessage:
    """A ``SystemMessage`` for ``text``, cache-marked where the provider allows.

    Anthropic prompt caching: a ``cache_control`` marker caches everything up
    to and including its block — the bound tool schemas plus this text — so
    the marker belongs only on a *stable* prompt (the base system prompt), not
    on per-turn context.
    """
    if settings.llm_provider.lower() == "anthropic":
        return SystemMessage(
            content=[
                {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}
            ]
        )
    return SystemMessage(content=text)
