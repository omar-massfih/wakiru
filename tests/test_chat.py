"""Chat core tests — the "undo" short-circuit and its upkeep gate.

No real graph or Codex call: a minimal stub stands in for the compiled agent,
and only ``run_chat``/``run_upkeep`` are exercised against a real (tmp_path)
calendar store, matching the style of test_calendar.py / test_undo.py.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from langchain_core.messages import AIMessageChunk

from assistant import chat
from assistant.calendar import context, ops, store
from assistant.config import Settings

THREAD = "telegram:1"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_auto_schedule=True,
        enable_write_confirmation=True,
        write_undo_window_minutes=15,
    )


def _iso_in(settings: Settings, **delta) -> str:
    return (context.now(settings) + timedelta(**delta)).isoformat(timespec="minutes")


class _FailingAgent:
    """An agent stub whose ``invoke`` fails the test if ever called."""

    def invoke(self, *args, **kwargs):
        pytest.fail("agent.invoke must not be called for an undo short-circuit")


class _CannedAgent:
    """An agent stub that returns a fixed reply, so ordinary turns can be observed."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.invoked = False

    def invoke(self, *args, **kwargs):
        self.invoked = True
        return {"messages": [type("Msg", (), {"content": self.reply})()]}


def _seed_booking(settings: Settings, monkeypatch) -> None:
    start = _iso_in(settings, days=2)
    canned = f'[{{"op": "create", "title": "Dentist", "start": "{start}"}}]'
    monkeypatch.setattr("assistant.calendar.ops.run_codex", lambda *a, **k: canned)
    ops.update_calendar(settings, "book the dentist friday", "Done!", THREAD)


# --- run_chat --------------------------------------------------------------- #


def test_run_chat_undo_short_circuits_without_invoking_agent(settings, monkeypatch) -> None:
    _seed_booking(settings, monkeypatch)
    reply = chat.run_chat(_FailingAgent(), "undo", THREAD, settings=settings)
    assert reply.startswith("Undone:")
    assert store.list_events(settings) == []


def test_run_chat_undo_variant_with_trailing_text(settings, monkeypatch) -> None:
    _seed_booking(settings, monkeypatch)
    reply = chat.run_chat(_FailingAgent(), "undo that", THREAD, settings=settings)
    assert reply.startswith("Undone:")


def test_run_chat_plain_message_goes_through_agent(settings) -> None:
    agent = _CannedAgent("hi there")
    reply = chat.run_chat(agent, "hello", THREAD, settings=settings)
    assert reply == "hi there"
    assert agent.invoked


def test_run_chat_undo_disabled_by_config_falls_through_to_agent(tmp_path) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"), enable_write_confirmation=False
    )
    agent = _CannedAgent("sure, undoing what?")
    reply = chat.run_chat(agent, "undo", THREAD, settings=settings)
    assert reply == "sure, undoing what?"
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


def test_run_chat_stream_undo_short_circuits(settings, monkeypatch) -> None:
    _seed_booking(settings, monkeypatch)
    agent = _StreamingAgent(["must not stream"])
    chunks = _collect(chat.run_chat_stream(agent, "undo", THREAD, settings=settings))
    assert len(chunks) == 1 and chunks[0].startswith("Undone:")
    assert not agent.streamed  # deterministic undo never touches the model
    assert store.list_events(settings) == []


# --- run_upkeep --------------------------------------------------------------- #


def test_run_upkeep_skips_all_upkeep_for_undo_turn(settings, monkeypatch) -> None:
    for name in ("update_memory", "maybe_summarize", "update_calendar"):
        monkeypatch.setattr(
            chat, name, lambda *a, **k: pytest.fail(f"{name} must not run for an undo turn")
        )
    chat.run_upkeep(_FailingAgent(), settings, "undo", "Undone: removed Dentist.", THREAD)


def test_run_upkeep_runs_calendar_for_normal_turn(settings, monkeypatch) -> None:
    called = {}
    monkeypatch.setattr(chat, "update_memory", lambda *a, **k: called.setdefault("memory", True))
    monkeypatch.setattr(chat, "maybe_summarize", lambda *a, **k: called.setdefault("summary", True))
    monkeypatch.setattr(
        chat, "update_calendar", lambda *a, **k: called.setdefault("calendar", True)
    )
    chat.run_upkeep(_FailingAgent(), settings, "book a dentist", "Done!", THREAD)
    assert called == {"memory": True, "summary": True, "calendar": True}
