"""Codex tool-calling shim tests — the protocol parser, prompt rendering, and
the hold-and-flush streaming filter. No subprocess: ``run_codex_stream`` is
monkeypatched where streaming is exercised.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from assistant.config import Settings
from assistant.llm import (
    ChatGptChatModel,
    CodexChatModel,
    _render_prompt,
    _soft_boundary,
    _split_leading_system,
    parse_tool_calls,
)
from assistant.tools import available_tools

_BLOCK = '```tool_call\n[{"name": "add_task", "arguments": {"title": "Buy milk"}}]\n```'


# --- parse_tool_calls -------------------------------------------------------- #


def test_parse_clean_block() -> None:
    content, calls = parse_tool_calls(_BLOCK)
    assert content == ""
    assert calls == [{"name": "add_task", "args": {"title": "Buy milk"}, "id": "call_1"}]


def test_parse_prose_then_block_keeps_prose_as_content() -> None:
    content, calls = parse_tool_calls("Adding that now.\n" + _BLOCK)
    assert content == "Adding that now."
    assert len(calls) == 1


def test_parse_multiple_calls_get_sequential_ids() -> None:
    block = (
        '```tool_call\n[{"name": "a", "arguments": {}},'
        ' {"name": "b", "arguments": {"x": 1}}]\n```'
    )
    _content, calls = parse_tool_calls(block)
    assert [c["id"] for c in calls] == ["call_1", "call_2"]
    assert calls[1]["args"] == {"x": 1}


def test_parse_single_object_and_args_variants() -> None:
    _c, calls = parse_tool_calls('```tool_call\n{"name": "a", "args": {"k": "v"}}\n```')
    assert calls[0]["args"] == {"k": "v"}
    # Arguments encoded as a JSON string still parse.
    _c, calls = parse_tool_calls(
        '```tool_call\n[{"name": "a", "arguments": "{\\"k\\": \\"v\\"}"}]\n```'
    )
    assert calls[0]["args"] == {"k": "v"}


def test_parse_bare_json_array_without_fence() -> None:
    _c, calls = parse_tool_calls('[{"name": "add_task", "arguments": {"title": "x"}}]')
    assert len(calls) == 1


def test_parse_malformed_fence_is_stripped_not_leaked() -> None:
    content, calls = parse_tool_calls("Sure.\n```tool_call\nnot json at all\n```")
    assert calls == []
    assert "tool_call" not in content and "not json" not in content
    assert content == "Sure."


def test_parse_plain_prose_untouched() -> None:
    text = "Here is some code:\n```python\nprint('hi')\n```\nEnjoy."
    content, calls = parse_tool_calls(text)
    assert calls == []
    assert content == text


# --- bind_tools + rendering --------------------------------------------------- #


def test_bind_tools_returns_copy_with_schemas() -> None:
    model = CodexChatModel(settings=Settings())
    schemas = [s.to_openai_tool() for s in available_tools(Settings())]
    bound = model.bind_tools(schemas)
    assert model.bound_tools == []  # the original stays clean
    assert len(bound.bound_tools) == len(schemas)
    assert bound.bound_tools[0]["function"]["name"]


def test_render_prompt_with_tools_includes_protocol_and_schemas() -> None:
    schemas = [s.to_openai_tool() for s in available_tools(Settings())]
    prompt = _render_prompt([HumanMessage(content="hi")], tools=schemas)
    assert "<tools>" in prompt and "```tool_call" in prompt
    assert '"add_task"' in prompt
    # And the plain path stays clean.
    assert "<tools>" not in _render_prompt([HumanMessage(content="hi")])


def test_render_prompt_replays_tool_history() -> None:
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="add milk"),
        AIMessage(
            content="",
            tool_calls=[{"name": "add_task", "args": {"title": "milk"}, "id": "call_1"}],
        ),
        ToolMessage(content="added task: milk", tool_call_id="call_1", name="add_task"),
    ]
    prompt = _render_prompt(messages)
    assert "```tool_call" in prompt  # the historical call is replayed verbatim
    assert "Tool result (add_task): added task: milk" in prompt


def test_split_leading_system_peels_consecutive_system_messages() -> None:
    messages = [
        SystemMessage(content="persona"),
        SystemMessage(content="context"),
        HumanMessage(content="hi"),
        SystemMessage(content="not leading"),
    ]
    system, rest = _split_leading_system(messages)
    assert system == "persona\n\ncontext"
    # Only the leading run is peeled; a later system message stays in the body.
    assert rest == messages[2:]


def test_split_leading_system_handles_no_system_messages() -> None:
    messages = [HumanMessage(content="hi")]
    system, rest = _split_leading_system(messages)
    assert system == "" and rest == messages


def test_chatgpt_prepare_routes_system_into_instructions() -> None:
    model = ChatGptChatModel(settings=Settings())
    messages = [
        SystemMessage(content="persona"),
        SystemMessage(content="context"),
        HumanMessage(content="hi there"),
    ]
    prompt, extra = model._prepare(messages, tools=None)
    assert extra == {"instructions": "persona\n\ncontext"}
    # The persona no longer rides in the user-turn text; the human turn does.
    assert "persona" not in prompt
    assert "User: hi there" in prompt


def test_chatgpt_prepare_no_system_falls_back_to_none() -> None:
    model = ChatGptChatModel(settings=Settings())
    _, extra = model._prepare([HumanMessage(content="hi")], tools=None)
    assert extra == {"instructions": None}


def test_codex_prepare_keeps_system_in_the_prompt() -> None:
    model = CodexChatModel(settings=Settings())
    messages = [SystemMessage(content="persona"), HumanMessage(content="hi")]
    prompt, extra = model._prepare(messages, tools=None)
    # Codex has no system slot: everything stays folded into the one prompt.
    assert extra == {}
    assert "System: persona" in prompt


def test_tool_schemas_stay_small() -> None:
    # Every enabled tool, email included: the prompt overhead must stay bounded.
    # (Raised from 9k when the mailbox-management tools landed — reply/archive/
    # mark-read/label plus the gated send_reply — and from 12k for
    # find_free_time, task recurrence, and attachment ingestion.)
    schemas = [
        s.to_openai_tool()
        for s in available_tools(Settings(enable_email=True, enable_email_send=True))
    ]
    assert len(json.dumps(schemas)) < 13_000


# --- streaming hold-and-flush -------------------------------------------------- #


def _stream_with(monkeypatch, deltas: list[str], bound: bool = True):
    model = CodexChatModel(settings=Settings())
    if bound:
        model = model.bind_tools(
            [s.to_openai_tool() for s in available_tools(Settings())]
        )
    monkeypatch.setattr(
        "assistant.llm.run_codex_stream", lambda prompt, settings=None: iter(deltas)
    )
    return list(model._stream([HumanMessage(content="hi")]))


def test_stream_fence_never_leaks_and_emits_tool_chunks(monkeypatch) -> None:
    chunks = _stream_with(monkeypatch, ["On it.\n``", "`tool_call\n[{\"name\": ", '"add_task", "arguments": {"title": "x"}}]\n```'])
    text = "".join(c.message.content for c in chunks)
    assert "tool_call" not in text and "{" not in text
    calls = [tc for c in chunks for tc in (c.message.tool_call_chunks or [])]
    assert len(calls) == 1 and calls[0]["name"] == "add_task"


def test_stream_plain_text_flushes_held_backticks(monkeypatch) -> None:
    chunks = _stream_with(monkeypatch, ["Use `uv sync`", " to install."])
    assert "".join(c.message.content for c in chunks) == "Use `uv sync` to install."


def test_stream_ordinary_code_fence_passes_through(monkeypatch) -> None:
    deltas = ["Here:\n```py", "thon\nprint('hi')\n``", "`\ndone"]
    chunks = _stream_with(monkeypatch, deltas)
    assert "".join(c.message.content for c in chunks) == "".join(deltas)


def test_stream_unbound_passes_through_untouched(monkeypatch) -> None:
    chunks = _stream_with(monkeypatch, ["a", "b"], bound=False)
    assert "".join(c.message.content for c in chunks) == "ab"


def test_soft_boundary_holds_only_fence_prefixes() -> None:
    assert _soft_boundary("hello ``", 0) == 6  # "``" could grow into the fence
    assert _soft_boundary("hello", 0) == 5
    assert _soft_boundary("x```toox", 0) == 8  # diverged — nothing to hold
