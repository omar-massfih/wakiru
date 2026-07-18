"""Chat core tests — every message is the model's turn, plus upkeep and errors.

No real graph or Codex call: a minimal stub stands in for the compiled agent,
and only ``run_chat``/``run_upkeep`` are exercised, matching the style of
test_calendar.py / test_undo.py.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk

from assistant import chat
from assistant.config import Settings

THREAD = "telegram:1"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_write_confirmation=True,
        write_undo_window_minutes=15,
    )


class _CannedAgent:
    """An agent stub that returns a fixed reply, so ordinary turns can be observed."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.invoked = False

    def invoke(self, *args, **kwargs):
        self.invoked = True
        return {"messages": [type("Msg", (), {"content": self.reply})()]}


# --- run_chat --------------------------------------------------------------- #


def test_run_chat_plain_message_goes_through_agent(settings) -> None:
    agent = _CannedAgent("hi there")
    reply = chat.run_chat(agent, "hello", THREAD, settings=settings)
    assert reply == "hi there"
    assert agent.invoked


def test_run_chat_undo_message_reaches_the_agent(settings) -> None:
    # "undo" is a normal turn now: the model interprets it and calls the
    # `undo` tool itself (see test_tools.py) — no keyword short-circuit.
    agent = _CannedAgent("Undone: removed Dentist.")
    reply = chat.run_chat(agent, "undo", THREAD, settings=settings)
    assert reply == "Undone: removed Dentist."
    assert agent.invoked


# --- run_chat_stream -------------------------------------------------------- #


class _StreamingAgent:
    """An agent stub whose ``astream`` yields message chunks in "messages" mode."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.streamed = False

    async def astream(self, _input, config=None, stream_mode=None):
        assert stream_mode == "messages"
        self.streamed = True
        for text in self.chunks:
            yield AIMessageChunk(content=text), {"langgraph_node": "codex"}


def _collect(aiter) -> list[str]:
    # No async pytest plugin here; drive the async generator via asyncio.run,
    # matching test_agent.py's ainvoke test.
    import asyncio

    async def _run() -> list[str]:
        return [chunk async for chunk in aiter]

    return asyncio.run(_run())


def test_run_chat_stream_yields_reply_chunks(settings) -> None:
    agent = _StreamingAgent(["Hel", "lo ", "there"])
    chunks = _collect(chat.run_chat_stream(agent, "hi", THREAD, settings=settings))
    assert chunks == ["Hel", "lo ", "there"]
    assert "".join(chunks) == "Hello there"
    assert agent.streamed


def test_run_chat_stream_skips_empty_chunks(settings) -> None:
    agent = _StreamingAgent(["", "hi", ""])
    chunks = _collect(chat.run_chat_stream(agent, "hi", THREAD, settings=settings))
    assert chunks == ["hi"]


def test_run_chat_stream_undo_message_streams_through_agent(settings) -> None:
    agent = _StreamingAgent(["Undone: removed Dentist."])
    chunks = _collect(chat.run_chat_stream(agent, "undo", THREAD, settings=settings))
    assert chunks == ["Undone: removed Dentist."]
    assert agent.streamed


# --- run_upkeep --------------------------------------------------------------- #


def test_run_upkeep_runs_memory_and_summary_for_normal_turn(settings, monkeypatch) -> None:
    called = {}
    monkeypatch.setattr(chat, "update_memory", lambda *a, **k: called.setdefault("memory", True))
    monkeypatch.setattr(chat, "maybe_summarize", lambda *a, **k: called.setdefault("summary", True))
    chat.run_upkeep(_CannedAgent(""), settings, "book a dentist", "Done!", THREAD)
    assert called == {"memory": True, "summary": True}


def test_run_upkeep_runs_for_undo_turn_too(settings, monkeypatch) -> None:
    # An undo turn is a turn like any other now — its upkeep runs.
    called = {}
    monkeypatch.setattr(chat, "update_memory", lambda *a, **k: called.setdefault("memory", True))
    monkeypatch.setattr(chat, "maybe_summarize", lambda *a, **k: called.setdefault("summary", True))
    chat.run_upkeep(_CannedAgent(""), settings, "undo", "Undone: removed Dentist.", THREAD)
    assert called == {"memory": True, "summary": True}


def test_run_chat_stream_filters_tool_call_chunks(settings) -> None:
    tool_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "add_task", "args": '{"title": "x"}', "id": "c1", "index": 0}
        ],
    )

    class _ToolStreamingAgent(_StreamingAgent):
        async def astream(self, _input, config=None, stream_mode=None):
            self.streamed = True
            yield tool_chunk, {"langgraph_node": "agent"}
            for text in self.chunks:
                yield AIMessageChunk(content=text), {"langgraph_node": "agent"}

    agent = _ToolStreamingAgent(["all ", "done"])
    chunks = _collect(chat.run_chat_stream(agent, "hi", THREAD, settings=settings))
    assert chunks == ["all ", "done"]  # the structured call never reaches the wire


def test_error_reply_distinguishes_failure_kinds() -> None:
    from assistant.codex_runner import CodexError, CodexTimeoutError

    timeout = chat.error_reply(CodexTimeoutError("Codex timed out after 300s."))
    snag = chat.error_reply(CodexError("Codex exited with code 1"))
    unexpected = chat.error_reply(ValueError("boom"))

    assert "too long" in timeout
    assert chat.error_reply(TimeoutError()) == timeout  # plain timeouts map alike
    assert "snag" in snag
    assert "unexpected" in unexpected.lower()
    # No internals leak into any of them.
    for text in (timeout, snag, unexpected):
        assert "Codex" not in text and "boom" not in text


def test_error_reply_maps_chatgpt_failures_like_codex() -> None:
    from assistant.chatgpt_backend import (
        ChatGptAuthError,
        ChatGptError,
        ChatGptTimeoutError,
    )
    from assistant.codex_runner import CodexError, CodexTimeoutError

    assert chat.error_reply(ChatGptTimeoutError("stalled")) == chat.error_reply(
        CodexTimeoutError("slow")
    )
    assert chat.error_reply(ChatGptError("HTTP 500")) == chat.error_reply(
        CodexError("code 1")
    )
    # Auth expiry is user-actionable, so it gets its own message.
    auth = chat.error_reply(ChatGptAuthError("refresh failed"))
    assert "codex login" in auth
    assert "HTTP" not in auth and "refresh failed" not in auth
