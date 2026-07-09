"""CLI channel tests — the stdin REPL over the shared chat core.

The agent build and the chat/upkeep calls are monkeypatched, so no Codex,
network, or graph is involved; only the loop's control flow is exercised.
"""

from __future__ import annotations

import builtins

from assistant import cli


def _run(monkeypatch, capsys, inputs, run_chat=None, upkeep=None):
    """Drive main() with a scripted stdin (raising EOFError when exhausted)."""
    it = iter(inputs)

    def fake_input(_prompt=""):
        try:
            return next(it)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(cli, "build_agent", lambda settings: object())
    monkeypatch.setattr(cli, "run_chat", run_chat or (lambda *a, **k: "reply"))
    monkeypatch.setattr(cli, "run_upkeep", upkeep or (lambda *a, **k: None))
    cli.main()
    return capsys.readouterr().out


def test_turn_prints_reply_and_runs_upkeep(monkeypatch, capsys) -> None:
    seen: list[tuple] = []
    out = _run(
        monkeypatch,
        capsys,
        ["hello"],
        run_chat=lambda agent, msg, thread, **k: f"echo:{msg} [{thread}]",
        upkeep=lambda agent, s, msg, reply, thread: seen.append((msg, reply, thread)),
    )
    assert "bot> echo:hello [cli:default]" in out
    assert seen == [("hello", "echo:hello [cli:default]", "cli:default")]


def test_blank_lines_are_skipped(monkeypatch, capsys) -> None:
    calls: list[str] = []
    _run(
        monkeypatch,
        capsys,
        ["", "  ", "hi"],
        run_chat=lambda agent, msg, thread, **k: calls.append(msg) or "ok",
    )
    assert calls == ["hi"]  # blank/whitespace lines never reach the model


def test_exit_quits_the_loop(monkeypatch, capsys) -> None:
    calls: list[str] = []
    _run(
        monkeypatch,
        capsys,
        ["exit", "should-not-run"],
        run_chat=lambda *a, **k: calls.append("ran") or "ok",
    )
    assert calls == []  # loop stops at "exit" before the next turn


def test_codex_error_is_reported_and_loop_continues(monkeypatch, capsys) -> None:
    from assistant.codex_runner import CodexError

    replies = iter([CodexError("boom"), "recovered"])

    def flaky(agent, msg, thread, **k):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    out = _run(monkeypatch, capsys, ["first", "second"], run_chat=flaky)
    assert "recovered" in out  # a failed turn doesn't kill the REPL
