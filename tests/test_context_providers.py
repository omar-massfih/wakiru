"""Context-provider registry tests — gating, isolation, and ordering."""

from __future__ import annotations

import pytest

from assistant.config import Settings
from assistant.context_providers import ContextProvider, build_context


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), timezone="Europe/Oslo")


def _provider(name: str, enabled: bool, text: str) -> ContextProvider:
    return ContextProvider(name, lambda s, e=enabled: e, lambda ctx, t=text: t)


def test_disabled_provider_is_omitted_entirely(settings) -> None:
    blocks = build_context(
        settings,
        "q",
        "t1",
        providers=[_provider("on", True, "hello"), _provider("off", False, "hidden")],
    )
    assert blocks == {"on": "hello"}


def test_failing_provider_contributes_empty_and_never_starves_others(settings) -> None:
    def boom(ctx):
        raise RuntimeError("subsystem down")

    blocks = build_context(
        settings,
        "q",
        "t1",
        providers=[
            ContextProvider("broken", lambda s: True, boom),
            _provider("healthy", True, "still here"),
        ],
    )
    assert blocks == {"broken": "", "healthy": "still here"}


def test_registry_order_is_block_order(settings) -> None:
    blocks = build_context(
        settings,
        "q",
        "t1",
        providers=[_provider(n, True, n) for n in ("b", "a", "c")],
    )
    assert list(blocks) == ["b", "a", "c"]


def test_provider_sees_the_turn(settings) -> None:
    seen = {}

    def capture(ctx):
        seen["query"] = ctx.query
        seen["thread_id"] = ctx.thread_id
        return ""

    build_context(
        settings, "find my notes", "telegram:7",
        providers=[ContextProvider("cap", lambda s: True, capture)],
    )
    assert seen == {"query": "find my notes", "thread_id": "telegram:7"}


def test_default_registry_gates_follow_settings(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )
    blocks = build_context(settings, "q", "t1")
    # Email is off by default; the always-on features contribute blocks.
    assert "mail" not in blocks
    assert {"recall", "profile", "agenda", "tasks"} <= set(blocks)
    assert "Current date and time" in blocks["agenda"]

    lean = settings.model_copy(update={"enable_calendar": False, "enable_tasks": False})
    lean_blocks = build_context(lean, "q", "t1")
    assert "agenda" not in lean_blocks and "tasks" not in lean_blocks
