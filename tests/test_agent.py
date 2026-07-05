"""Smoke tests — no real Codex invocation."""

from __future__ import annotations

from fastapi.testclient import TestClient

import pytest

from assistant.api import app
from assistant.codex_runner import build_command
from assistant.config import Settings
from assistant.llm import CodexChatModel, build_model


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_agent_graph_compiles(tmp_path) -> None:
    from assistant.agent import build_agent

    graph = build_agent(Settings(memory_dir=str(tmp_path / "memory")))
    # A compiled graph exposes invoke; we don't call it (that would hit Codex).
    assert hasattr(graph, "invoke")


def test_build_command_defaults() -> None:
    cmd = build_command("hello", "/tmp/out.txt", Settings())
    assert cmd[:2] == ["codex", "exec"]
    assert "--skip-git-repo-check" in cmd
    assert cmd[-1] == "hello"  # prompt is the trailing positional arg
    assert "-o" in cmd and "/tmp/out.txt" in cmd
    assert "-s" in cmd and "read-only" in cmd


def test_build_command_includes_model_and_cwd() -> None:
    settings = Settings(codex_model="gpt-5-codex", codex_working_dir="/work")
    cmd = build_command("hi", "/tmp/o.txt", settings)
    assert "-m" in cmd and "gpt-5-codex" in cmd
    assert "-C" in cmd and "/work" in cmd


def test_build_model_defaults_to_codex() -> None:
    model = build_model(Settings())
    assert isinstance(model, CodexChatModel)


def test_build_model_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_model(Settings(llm_provider="does-not-exist"))


def test_stub_providers_raise_not_implemented() -> None:
    for provider in ("openai", "anthropic"):
        with pytest.raises(NotImplementedError):
            build_model(Settings(llm_provider=provider))
