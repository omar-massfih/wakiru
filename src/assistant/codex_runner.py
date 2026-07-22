"""Thin subprocess wrapper around ``codex exec`` (non-interactive Codex CLI).

Codex is itself an autonomous agent (its own model, tools, and sandbox). We drive
it programmatically and capture its final message. Auth is whatever ``codex login``
established (e.g. ChatGPT sign-in) — no API key is passed here.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# Cap on concurrent Codex subprocesses (see run_codex). Sized from the first
# Settings that reaches run_codex; one process-wide semaphore is enough because
# settings are effectively a singleton.
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_lock = threading.Lock()


def _codex_slot(settings: Settings) -> threading.BoundedSemaphore:
    global _semaphore
    with _semaphore_lock:
        if _semaphore is None:
            _semaphore = threading.BoundedSemaphore(max(settings.codex_max_concurrency, 1))
        return _semaphore


class CodexError(RuntimeError):
    """Raised when the Codex CLI exits non-zero or times out."""


class CodexTimeoutError(CodexError):
    """Raised when a Codex invocation exceeds ``codex_timeout``.

    A subclass so ``except CodexError`` callers keep working, while channels
    can tell "took too long" from "broke" when explaining a failure.
    """


def build_command(
    output_file: str, settings: Settings, json_events: bool = False
) -> list[str]:
    """Assemble the ``codex exec`` argv. Kept pure so it can be unit-tested.

    The prompt itself is NOT part of the argv: it is piped on stdin (the ``-``
    positional). A long conversation flattened into a single argument would hit
    the kernel's per-argument size limit (~128 KB on Linux) and fail the exec.

    ``json_events=True`` adds ``--json`` so Codex prints its event stream as
    JSONL on stdout (used by :func:`run_codex_stream`); ``-o`` still captures
    the final message either way.
    """
    cmd: list[str] = [settings.codex_bin]
    if settings.codex_web_search:
        # Must precede the `exec` subcommand — codex rejects it after.
        cmd.append("--search")
    cmd += [
        "exec",
        "--skip-git-repo-check",
        "--color",
        "never",
        "-s",
        settings.codex_sandbox,
        "-o",
        output_file,
    ]
    if json_events:
        cmd.append("--json")
    if settings.codex_model:
        cmd += ["-m", settings.codex_model]
    if settings.codex_working_dir:
        cmd += ["-C", settings.codex_working_dir]
    # Read the prompt from stdin.
    cmd.append("-")
    return cmd


def run_codex(prompt: str, settings: Settings | None = None) -> str:
    """Run one non-interactive Codex turn and return its final message text.

    Concurrency is bounded by ``codex_max_concurrency``: one chat turn fans out
    into several Codex calls (reply, then memory/calendar/summary upkeep), each
    of which can block a threadpool worker for up to ``codex_timeout`` seconds —
    unbounded, a small burst could saturate the server's worker pool. Excess
    calls simply queue for a slot.
    """
    settings = settings or get_settings()

    with _codex_slot(settings), tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "last_message.txt"
        cmd = build_command(str(out_path), settings)

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=settings.codex_timeout,
            )
        except FileNotFoundError as exc:
            raise CodexError(
                f"Codex binary {settings.codex_bin!r} not found on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CodexTimeoutError(
                f"Codex timed out after {settings.codex_timeout}s."
            ) from exc

        if result.returncode != 0:
            raise CodexError(
                f"Codex exited with code {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        if out_path.exists():
            message = out_path.read_text(encoding="utf-8").strip()
            if message:
                return message

        # Fallback: Codex wrote nothing to the last-message file. stdout is
        # usually agent/log chatter rather than a clean reply — surface it.
        logger.warning(
            "codex wrote no final message to -o; falling back to stdout (%d chars)",
            len(result.stdout),
        )
        return result.stdout.strip()


class CodexStreamParser:
    """Reduce ``codex exec --json`` stdout lines to user-visible text deltas.

    Depending on the CLI version the text of an ``agent_message`` item lands as
    growing snapshots (``item.updated``) or one whole ``item.completed`` — both
    shapes reduce to increments here, with ``"\\n\\n"`` inserted between distinct
    message items. Failure events are collected on :attr:`failure` rather than
    raised, so the process owner decides how to surface them after the stream.
    """

    def __init__(self) -> None:
        self.emitted = ""  # text already emitted for the current agent_message item
        self.item_id: str | None = None
        self.failure: str | None = None

    def feed(self, line: str) -> list[str]:
        """The deltas one stdout line unlocks (often none)."""
        try:
            event = json.loads(line)
        except ValueError:
            return []  # non-JSON chatter interleaved on stdout
        etype = event.get("type", "")
        if etype in ("item.updated", "item.completed"):
            item = event.get("item") or {}
            if item.get("type") != "agent_message":
                return []
            deltas: list[str] = []
            text = item.get("text") or ""
            if item.get("id") != self.item_id:  # a new message item begins
                if self.emitted:
                    deltas.append("\n\n")
                self.item_id = item.get("id")
                self.emitted = ""
            if text.startswith(self.emitted):
                delta = text[len(self.emitted) :]
            else:  # snapshot diverged from what we sent — resync whole
                delta = ("\n" if self.emitted else "") + text
            self.emitted = text
            if delta:
                deltas.append(delta)
            return deltas
        if etype == "turn.failed":
            self.failure = (event.get("error") or {}).get("message") or self.failure
        elif etype == "error":
            self.failure = self.failure or event.get("message")
        return []


class _StreamingCodexProcess:
    """One streaming ``codex exec --json`` subprocess and its plumbing threads.

    Feeds stdin and drains stderr on daemon threads so neither pipe can deadlock
    against the owner's blocking reads of stdout, and arms a watchdog Timer
    instead of ``subprocess.run(timeout=)`` so those reads stay blocking and
    chunks flow the moment codex prints them.
    """

    def __init__(self, cmd: list[str], prompt: str, timeout: float, codex_bin: str) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise CodexError(f"Codex binary {codex_bin!r} not found on PATH.") from exc

        self.timed_out = threading.Event()
        self._stderr_text: list[str] = []

        def _feed_stdin() -> None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(prompt)
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):  # codex exited first
                pass

        def _drain_stderr() -> None:
            assert self.proc.stderr is not None
            self._stderr_text.append(self.proc.stderr.read())

        threading.Thread(target=_feed_stdin, daemon=True).start()
        self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()

        def _expire() -> None:
            self.timed_out.set()
            self.proc.kill()

        self._watchdog = threading.Timer(timeout, _expire)
        self._watchdog.start()

    def kill(self) -> None:
        self.proc.kill()

    def reap(self) -> None:
        self._watchdog.cancel()
        try:
            # Normal path: stdout hit EOF because codex is exiting; killed
            # paths (watchdog / GeneratorExit) reap immediately.
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def stderr(self) -> str:
        # Join the drain thread before reading: reap() only wait()s the process,
        # which leaves the daemon's final stderr.read()/append racing the error
        # path that calls this — without the join the detail can come back empty.
        self._stderr_thread.join(timeout=5)
        return (self._stderr_text[0].strip() if self._stderr_text else "") or ""


def run_codex_stream(prompt: str, settings: Settings | None = None) -> Iterator[str]:
    """Run one Codex turn and yield the reply text incrementally.

    Drives ``codex exec --json`` and reduces its JSONL events to text deltas via
    :class:`CodexStreamParser`; worst case the caller gets the full reply in a
    single chunk (never less than the non-streaming path). Error semantics match
    :func:`run_codex`: any failure raises :class:`CodexError`, possibly after
    some text has been yielded.

    The concurrency slot is held until the generator is exhausted or closed;
    closing it early kills the subprocess.
    """
    settings = settings or get_settings()

    with _codex_slot(settings), tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "last_message.txt"
        cmd = build_command(str(out_path), settings, json_events=True)
        runner = _StreamingCodexProcess(
            cmd, prompt, settings.codex_timeout, settings.codex_bin
        )
        parser = CodexStreamParser()
        try:
            assert runner.proc.stdout is not None
            for line in runner.proc.stdout:
                yield from parser.feed(line)
        except GeneratorExit:  # consumer stopped iterating — don't leave codex running
            runner.kill()
            raise
        finally:
            runner.reap()

        if runner.timed_out.is_set():
            raise CodexTimeoutError(f"Codex timed out after {settings.codex_timeout}s.")
        if parser.failure or runner.proc.returncode != 0:
            raise CodexError(
                parser.failure
                or f"Codex exited with code {runner.proc.returncode}: {runner.stderr()}"
            )

        if not parser.emitted and out_path.exists():
            # No agent_message events surfaced (schema drift?) — fall back to
            # the -o file so the caller still gets the reply, in one chunk.
            message = out_path.read_text(encoding="utf-8").strip()
            if message:
                logger.warning("codex --json yielded no message events; using -o file")
                yield message
