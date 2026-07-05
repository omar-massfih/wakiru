"""Thin subprocess wrapper around ``codex exec`` (non-interactive Codex CLI).

Codex is itself an autonomous agent (its own model, tools, and sandbox). We drive
it programmatically and capture its final message. Auth is whatever ``codex login``
established (e.g. ChatGPT sign-in) — no API key is passed here.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .config import Settings, get_settings


class CodexError(RuntimeError):
    """Raised when the Codex CLI exits non-zero or times out."""


def build_command(prompt: str, output_file: str, settings: Settings) -> list[str]:
    """Assemble the ``codex exec`` argv. Kept pure so it can be unit-tested."""
    cmd: list[str] = [
        settings.codex_bin,
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
    # Prompt is the trailing positional argument.
    cmd.append(prompt)
    return cmd


def run_codex(prompt: str, settings: Settings | None = None) -> str:
    """Run one non-interactive Codex turn and return its final message text."""
    settings = settings or get_settings()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "last_message.txt"
        cmd = build_command(prompt, str(out_path), settings)

        try:
            result = subprocess.run(
                cmd,
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

        # Fallback: Codex wrote nothing to the last-message file.
        return result.stdout.strip()
