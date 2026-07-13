"""Smoke tests — no real Codex invocation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
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


def test_build_command_json_events_flag() -> None:
    cmd = build_command("/tmp/o.txt", Settings(), json_events=True)
    assert "--json" in cmd
    assert "--json" not in build_command("/tmp/o.txt", Settings())


# --- run_codex_stream (fake codex binary emitting JSONL) --------------------- #


def _fake_codex(tmp_path, body: str) -> Settings:
    """A stand-in ``codex`` executable; ``body`` is its Python source."""
    script = tmp_path / "fake-codex"
    script.write_text("#!/usr/bin/env python3\nimport json, sys\nsys.stdin.read()\n" + body)
    script.chmod(0o755)
    return Settings(codex_bin=str(script))


def test_run_codex_stream_yields_increments(tmp_path) -> None:
    from assistant.codex_runner import run_codex_stream

    settings = _fake_codex(
        tmp_path,
        """
events = [
    {"type": "thread.started", "thread_id": "t1"},
    {"type": "turn.started"},
    {"type": "item.updated", "item": {"id": "m1", "type": "agent_message", "text": "Hel"}},
    {"type": "item.updated", "item": {"id": "m1", "type": "agent_message", "text": "Hello"}},
    {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "Hello world"}},
    {"type": "turn.completed"},
]
for e in events:
    print(json.dumps(e))
""",
    )
    assert list(run_codex_stream("hi", settings=settings)) == ["Hel", "lo", " world"]


def test_run_codex_stream_separates_message_items(tmp_path) -> None:
    from assistant.codex_runner import run_codex_stream

    settings = _fake_codex(
        tmp_path,
        """
events = [
    {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "first"}},
    {"type": "item.completed", "item": {"id": "m2", "type": "agent_message", "text": "second"}},
]
for e in events:
    print(json.dumps(e))
""",
    )
    assert "".join(run_codex_stream("hi", settings=settings)) == "first\n\nsecond"


def test_run_codex_stream_turn_failed_raises(tmp_path) -> None:
    from assistant.codex_runner import CodexError, run_codex_stream

    settings = _fake_codex(
        tmp_path,
        """
print(json.dumps({"type": "turn.failed", "error": {"message": "usage limit hit"}}))
""",
    )
    with pytest.raises(CodexError, match="usage limit hit"):
        list(run_codex_stream("hi", settings=settings))


def test_run_codex_stream_nonzero_exit_raises(tmp_path) -> None:
    from assistant.codex_runner import CodexError, run_codex_stream

    settings = _fake_codex(tmp_path, "sys.exit(3)\n")
    with pytest.raises(CodexError, match="code 3"):
        list(run_codex_stream("hi", settings=settings))


def test_run_codex_stream_falls_back_to_output_file(tmp_path) -> None:
    from assistant.codex_runner import run_codex_stream

    # No agent_message events at all — the -o file is the only reply source.
    settings = _fake_codex(
        tmp_path,
        """
out = sys.argv[sys.argv.index("-o") + 1]
open(out, "w").write("from the file")
print(json.dumps({"type": "turn.completed"}))
""",
    )
    assert list(run_codex_stream("hi", settings=settings)) == ["from the file"]


def test_run_codex_stream_timeout_kills_process(tmp_path) -> None:
    from assistant.codex_runner import CodexError, run_codex_stream

    settings = _fake_codex(tmp_path, "import time\ntime.sleep(30)\n")
    settings.codex_timeout = 1
    with pytest.raises(CodexError, match="timed out"):
        list(run_codex_stream("hi", settings=settings))


def test_codex_chat_model_stream_emits_chunks(monkeypatch) -> None:
    from assistant import llm as llm_module

    monkeypatch.setattr(
        llm_module, "run_codex_stream", lambda prompt, settings=None: iter(["a", "b"])
    )
    model = CodexChatModel(settings=Settings())
    # langchain appends a final empty metadata chunk; consumers filter empties.
    chunks = [c.content for c in model.stream([HumanMessage(content="hi")]) if c.content]
    assert chunks == ["a", "b"]


def test_build_model_defaults_to_codex() -> None:
    model = build_model(Settings())
    assert isinstance(model, CodexChatModel)


def test_build_model_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_model(Settings(llm_provider="does-not-exist"))


def test_api_providers_build_with_key() -> None:
    # ChatOpenAI / ChatAnthropic are constructed (no network); assert the
    # selected model and key flow through, and the default model applies.
    openai = build_model(Settings(llm_provider="openai", llm_api_key="sk-test"))
    assert openai.model_name == "gpt-4o"

    anthropic = build_model(
        Settings(llm_provider="anthropic", llm_api_key="sk-test", llm_model="claude-sonnet-5")
    )
    assert anthropic.model == "claude-sonnet-5"


def test_api_providers_require_key() -> None:
    for provider in ("openai", "anthropic"):
        with pytest.raises(ValueError, match="requires LLM_API_KEY"):
            build_model(Settings(llm_provider=provider))


def test_api_providers_use_llm_timeout_and_max_tokens() -> None:
    settings = Settings(llm_api_key="sk-test", llm_timeout=42, llm_max_tokens=1234)
    anthropic = build_model(settings.model_copy(update={"llm_provider": "anthropic"}))
    assert anthropic.max_tokens == 1234
    openai = build_model(settings.model_copy(update={"llm_provider": "openai"}))
    assert openai.max_tokens == 1234


def test_complete_text_goes_through_the_configured_provider(monkeypatch) -> None:
    from langchain_core.messages import AIMessage

    from assistant import llm as llm_module

    class Echo:
        def invoke(self, messages):
            return AIMessage(content=f"echo: {messages[0].content}")

    monkeypatch.setattr(llm_module, "build_model", lambda s=None: Echo())
    assert llm_module.complete_text("hi", Settings()) == "echo: hi"


def test_cacheable_system_message_marks_only_anthropic() -> None:
    from assistant.llm import cacheable_system_message

    plain = cacheable_system_message("base", Settings(llm_provider="codex"))
    assert plain.content == "base"

    marked = cacheable_system_message(
        "base", Settings(llm_provider="anthropic", llm_api_key="sk-test")
    )
    assert marked.content[0]["cache_control"] == {"type": "ephemeral"}
    assert marked.content[0]["text"] == "base"


def test_reply_prompt_carries_the_persona(monkeypatch, tmp_path) -> None:
    """The persistent persona block must lead every reply-path prompt."""
    from assistant import persona
    from assistant.agent import build_agent
    from assistant.chat import run_chat

    seen: dict[str, str] = {}

    def fake_run_codex(prompt, settings=None):
        seen["prompt"] = prompt
        return "hei"

    monkeypatch.setattr("assistant.llm.run_codex", fake_run_codex)
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_memory=False,
        enable_auto_memory=False,
        enable_calendar=False,
        enable_tasks=False,
        enable_docs=False,
    )
    agent = build_agent(settings)
    run_chat(agent, "hello", "t-prompt", settings=settings)
    assert persona.system_prompt(settings).splitlines()[0] in seen["prompt"]


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
    # Tool-less graph: FakeListChatModel has no bind_tools, and this test is
    # about summarization, not the tool loop.
    monkeypatch.setattr("assistant.agent.available_tools", lambda s: [])
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


# --- tool loop ---------------------------------------------------------------- #


def _fake_embed_patch(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )


class _ScriptedToolModel:
    """A fake chat model that emits scripted tool calls, tracking bind state."""

    def __init__(self, always_call: bool = False) -> None:
        self.always_call = always_call
        self.invocations: list[bool] = []  # True = the tools-bound copy was used
        self.bound_schemas: list[dict] = []

    def bind_tools(self, tools):
        self.bound_schemas = list(tools)
        parent = self

        class _Bound:
            def invoke(self, messages):
                return parent._invoke(bound=True)

        return _Bound()

    def invoke(self, messages):
        return self._invoke(bound=False)

    def _invoke(self, bound: bool):
        self.invocations.append(bound)
        wants_call = bound and (self.always_call or len(self.invocations) == 1)
        if wants_call:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "add_task",
                        "args": {"title": f"task-{len(self.invocations)}"},
                        "id": "call_1",
                    }
                ],
            )
        return AIMessage(content="Done — added to your list.")


def _build_tool_agent(tmp_path, monkeypatch, model, **overrides):
    from assistant.agent import build_agent

    _fake_embed_patch(monkeypatch)
    monkeypatch.setattr("assistant.agent.build_model", lambda s=None: model)
    settings = Settings(memory_dir=str(tmp_path / "memory"), **overrides)
    return build_agent(settings), settings


def test_tool_loop_executes_call_then_replies(tmp_path, monkeypatch) -> None:
    from assistant.chat import run_chat
    from assistant.tasks import store as tasks_store

    model = _ScriptedToolModel()
    agent, settings = _build_tool_agent(tmp_path, monkeypatch, model)
    reply = run_chat(agent, "add a task", "t1", settings=settings)

    assert reply == "Done — added to your list."
    assert model.invocations == [True, True]  # one tool round, then the reply
    assert [t.title for t in tasks_store.list_tasks(settings)] == ["task-1"]
    assert {s["function"]["name"] for s in model.bound_schemas} >= {"add_task", "remember"}

    history = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert [m.type for m in history] == ["human", "ai", "tool", "ai"]


def test_tool_loop_cap_forces_plain_reply(tmp_path, monkeypatch) -> None:
    from assistant.chat import run_chat

    model = _ScriptedToolModel(always_call=True)
    agent, settings = _build_tool_agent(
        tmp_path, monkeypatch, model, tool_max_rounds=2
    )
    reply = run_chat(agent, "go wild", "t1", settings=settings)

    assert reply == "Done — added to your list."
    # Bound passes until the cap, then exactly one unbound (tool-less) pass.
    assert model.invocations[-1] is False
    assert all(model.invocations[:-1])

    history = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert not getattr(history[-1], "tool_calls", None)  # no dangling calls
    assert any(
        m.type == "tool" and "budget exhausted" in str(m.content).lower()
        for m in history
    )


def test_empty_tool_registry_binds_nothing(tmp_path, monkeypatch) -> None:
    from assistant.chat import run_chat

    model = _ScriptedToolModel()
    monkeypatch.setattr("assistant.agent.available_tools", lambda s: [])
    agent, settings = _build_tool_agent(tmp_path, monkeypatch, model)
    reply = run_chat(agent, "hello", "t1", settings=settings)
    assert model.bound_schemas == []  # nothing registered: the model is never bound
    assert model.invocations == [False]
    assert reply == "Done — added to your list."


def test_unknown_tool_call_gets_error_tool_message(tmp_path, monkeypatch) -> None:
    from assistant.chat import run_chat

    class _UnknownToolModel(_ScriptedToolModel):
        def _invoke(self, bound: bool):
            self.invocations.append(bound)
            if bound and len(self.invocations) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "launch_rocket", "args": {}, "id": "c1"}],
                )
            return AIMessage(content="Sorry, I can't do that.")

    model = _UnknownToolModel()
    agent, settings = _build_tool_agent(tmp_path, monkeypatch, model)
    run_chat(agent, "launch it", "t1", settings=settings)
    history = agent.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    tool_msgs = [m for m in history if m.type == "tool"]
    assert len(tool_msgs) == 1 and "Unknown tool" in str(tool_msgs[0].content)


# --- expanded recall query ------------------------------------------------------ #


def test_expanded_recall_query_carries_recent_context(tmp_path) -> None:
    from assistant.agent import expanded_recall_query

    settings = Settings(memory_dir=str(tmp_path / "memory"))
    messages = [
        HumanMessage(content="Tell me about the Bergen trip"),
        AIMessage(content="You leave Friday morning and stay two nights."),
        HumanMessage(content="when is it?"),
    ]
    query = expanded_recall_query(messages, "Planning a work trip.", settings)
    assert query.startswith("when is it?")  # the live turn leads
    assert "Bergen" in query
    assert "Planning a work trip." in query


def test_expanded_recall_query_disabled_is_latest_only(tmp_path) -> None:
    from assistant.agent import expanded_recall_query

    settings = Settings(
        memory_dir=str(tmp_path / "memory"), recall_context_extra_chars=0
    )
    messages = [
        HumanMessage(content="Tell me about Bergen"),
        AIMessage(content="It rains."),
        HumanMessage(content="when?"),
    ]
    assert expanded_recall_query(messages, "summary text", settings) == "when?"


# --- summarize_fold with tool messages ------------------------------------------ #


def test_summarize_fold_never_orphans_a_tool_message(tmp_path) -> None:
    from langchain_core.messages import ToolMessage

    from assistant.agent import summarize_fold

    settings = _wm_settings(tmp_path)  # max=4, keep_recent=2
    messages = [
        *_history(4),
        ToolMessage(content="added task: x", tool_call_id="c1", id="4"),
        AIMessage(content="done", id="5"),
    ]
    update = summarize_fold(settings, FakeListChatModel(responses=["sum"]), messages, "")
    assert update is not None
    removed = {m.id for m in update["messages"]}
    # The boundary slides past the ToolMessage so the kept tail can't open on one.
    assert "4" in removed
    assert removed == {"0", "1", "2", "3", "4"}
