"""Thin subprocess wrapper around ``codex exec`` (non-interactive Codex CLI).

Codex is itself an autonomous agent (its own model, tools, and sandbox). We drive
it programmatically and capture its final message. Auth is whatever ``codex login``
established (e.g. ChatGPT sign-in) — no API key is passed here.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
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


def build_command(output_file: str, settings: Settings) -> list[str]:
    """Assemble the ``codex exec`` argv. Kept pure so it can be unit-tested.

    The prompt itself is NOT part of the argv: it is piped on stdin (the ``-``
    positional). A long conversation flattened into a single argument would hit
    the kernel's per-argument size limit (~128 KB on Linux) and fail the exec.
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
            raise CodexError(
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
