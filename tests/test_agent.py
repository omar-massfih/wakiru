"""Smoke tests — no real Codex invocation."""

from __future__ import annotations

from fastapi.testclient import TestClient

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

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
    cmd = build_command("/tmp/out.txt", Settings())
    assert cmd[:2] == ["codex", "exec"]
    assert "--skip-git-repo-check" in cmd
    assert cmd[-1] == "-"  # prompt arrives on stdin, not argv (argv size limits)
    assert "-o" in cmd and "/tmp/out.txt" in cmd
    assert "-s" in cmd and "read-only" in cmd


def test_build_command_includes_model_and_cwd() -> None:
    settings = Settings(codex_model="gpt-5-codex", codex_working_dir="/work")
    cmd = build_command("/tmp/o.txt", settings)
    assert "-m" in cmd and "gpt-5-codex" in cmd
    assert "-C" in cmd and "/work" in cmd


def test_build_command_web_search_precedes_exec() -> None:
    cmd = build_command("/tmp/o.txt", Settings(codex_web_search=True))
    assert cmd[:3] == ["codex", "--search", "exec"]


def test_run_codex_pipes_prompt_on_stdin(monkeypatch) -> None:
    from assistant import codex_runner

    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs.get("input")

        class Result:
            returncode = 0
            stdout = "final message"
            stderr = ""

        return Result()

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    long_prompt = "x" * 500_000  # would exceed the kernel's per-arg limit as argv
    assert codex_runner.run_codex(long_prompt, settings=Settings()) == "final message"
    assert seen["input"] == long_prompt
    assert long_prompt not in seen["cmd"]


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


# --- background working-memory summarization -------------------------------- #


def _wm_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_calendar=False,
        working_memory_max_messages=4,
        working_memory_keep_recent=2,
        **overrides,
    )


def _history(n: int) -> list:
    return [
        (HumanMessage if i % 2 == 0 else AIMessage)(content=f"m{i}", id=str(i))
        for i in range(n)
    ]


def test_summarize_fold_below_threshold_returns_none(tmp_path) -> None:
    from assistant.agent import summarize_fold

    settings = _wm_settings(tmp_path)
    model = FakeListChatModel(responses=["unused"])
    assert summarize_fold(settings, model, _history(4), "") is None


def test_summarize_fold_folds_older_messages(tmp_path) -> None:
    from assistant.agent import summarize_fold

    settings = _wm_settings(tmp_path)
    model = FakeListChatModel(responses=["folded summary"])
    update = summarize_fold(settings, model, _history(6), "old summary")
    assert update is not None
    assert update["summary"] == "folded summary"
    assert all(isinstance(m, RemoveMessage) for m in update["messages"])
    # Everything but the keep_recent tail is folded, by id.
    assert {m.id for m in update["messages"]} == {"0", "1", "2", "3"}


def test_summarize_fold_model_failure_returns_none(tmp_path) -> None:
    from assistant.agent import summarize_fold

    class BoomModel:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("model down")

    settings = _wm_settings(tmp_path)
    assert summarize_fold(settings, BoomModel(), _history(6), "") is None


def test_maybe_summarize_trims_thread_in_background(tmp_path, monkeypatch) -> None:
    from assistant.agent import build_agent, maybe_summarize

    # Offline: fake both the chat model and the embedder.
    monkeypatch.setattr(
        "assistant.agent.build_model",
        lambda s=None: FakeListChatModel(responses=["ok"]),
    )
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )
    settings = _wm_settings(tmp_path)
    graph = build_agent(settings)
    config = {"configurable": {"thread_id": "t1"}}
    for i in range(3):
        graph.invoke({"messages": [HumanMessage(content=f"message {i}")]}, config=config)

    # The reply path no longer trims: 6 messages sit above the threshold of 4.
    assert len(graph.get_state(config).values["messages"]) == 6

    maybe_summarize(graph, settings, "t1")
    state = graph.get_state(config)
    assert len(state.values["messages"]) == 2  # keep_recent tail, fully folded
    assert state.values["summary"]


def test_run_codex_bounds_concurrency(monkeypatch) -> None:
    """N parallel calls never run more than codex_max_concurrency subprocesses."""
    import threading
    import time
    from types import SimpleNamespace

    from assistant import codex_runner

    monkeypatch.setattr(codex_runner, "_semaphore", None)  # fresh, sized from these settings
    settings = Settings(codex_max_concurrency=2)

    active = 0
    peak = 0
    gauge = threading.Lock()

    def slow_fake_run(cmd, **kwargs):
        nonlocal active, peak
        with gauge:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with gauge:
            active -= 1
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(codex_runner.subprocess, "run", slow_fake_run)

    threads = [
        threading.Thread(target=codex_runner.run_codex, args=("hi", settings))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak <= 2


def test_ainvoke_runs_codex_without_blocking_the_loop(monkeypatch) -> None:
    import asyncio

    from assistant import llm as llm_module

    monkeypatch.setattr(llm_module, "run_codex", lambda prompt, settings=None: "async-svar")
    model = CodexChatModel(settings=Settings())
    result = asyncio.run(model.ainvoke([HumanMessage(content="hei")]))
    assert result.content == "async-svar"
