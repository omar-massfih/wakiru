"""Run short Python scripts the model writes, in a hardened in-container subprocess.

This is the executor behind the ``run_python`` tool. The model writes a script to
compute over data it has already surfaced (document text, an attachment, calendar
or task data) and we hand back its stdout so the reply can use or explain it.

Hardening is defense-in-depth, not a security boundary — the script still runs in
the assistant's own container, so it can *read* the container filesystem. What we
do close off is the highest-value leak and the obvious footguns:

* **Stripped environment** — the child gets a minimal ``PATH``/``LANG``/``TMPDIR``
  allowlist, never the parent's ``os.environ`` where API tokens and OAuth secrets
  live. This is the single most important mitigation.
* **No network** — a preamble neuters ``socket`` before the script runs. A
  determined script could still reach the network by other means; like
  :mod:`assistant.netguard`, this is defense-in-depth, not a hard guarantee.
* **Resource limits** — CPU, address space, and file-size rlimits (POSIX) plus a
  wall-clock timeout bound runaway scripts.
* **Isolated interpreter** — ``python -I -S`` ignores ``PYTHONPATH`` / user site /
  site customization, and the cwd is a throwaway temp dir.

If stronger isolation is ever needed, move execution to a locked-down executor
(namespaces / nsjail / a throwaway container) behind :func:`run_python`'s
signature — no tool or agent changes required.

Concurrency is bounded exactly like :mod:`assistant.codex_runner`: one chat turn
can fan out into several subprocesses, each holding a worker thread until it
finishes, so a shared semaphore queues the excess.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# Sized from the first Settings that reaches run_python; one process-wide
# semaphore is enough because settings are effectively a singleton.
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_lock = threading.Lock()


def _exec_slot(settings: Settings) -> threading.BoundedSemaphore:
    global _semaphore
    with _semaphore_lock:
        if _semaphore is None:
            _semaphore = threading.BoundedSemaphore(
                max(settings.code_exec_max_concurrency, 1)
            )
        return _semaphore


# Prepended to every script: block network as defense-in-depth before any of the
# model's code runs. Kept tiny and dependency-free so it never masks a real error.
_PREAMBLE = """\
import socket as _socket


def _blocked(*_args, **_kwargs):
    raise OSError("network access is disabled in this sandbox")


_socket.socket = _blocked
_socket.create_connection = _blocked
_socket.create_server = _blocked
del _socket
"""


def _rlimits(settings: Settings):
    """A POSIX ``preexec_fn`` that caps CPU, memory, and file size, or None.

    Returns None off POSIX (no ``fork``/``resource``) so the subprocess still
    runs — the wall-clock timeout and env stripping remain in force there.
    """
    import os

    if not hasattr(os, "fork"):
        return None
    try:
        import resource
    except ImportError:  # non-POSIX
        return None

    cpu_seconds = max(settings.code_exec_timeout, 1) + 1
    address_space = max(settings.code_exec_max_memory_mb, 1) * 1024 * 1024
    # Cap scratch writes generously above the output cap so a legitimate temp
    # file is fine but a disk-filling loop is not.
    file_size = max(settings.code_exec_max_output_chars * 8, 16 * 1024 * 1024)

    limits = (
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_AS, address_space),
        (resource.RLIMIT_FSIZE, file_size),
    )

    import contextlib

    def apply() -> None:
        # Each limit is best-effort: some (RLIMIT_AS on macOS) aren't
        # enforceable everywhere, and a raise here would kill the child in the
        # fork before exec. On the Linux container all three apply.
        for what, value in limits:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(what, (value, value))

    return apply


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated at {limit} characters]"


def _format_result(
    stdout: str, stderr: str, returncode: int, limit: int
) -> str:
    """Fold a finished run into the one string the model reads back.

    An exception in the script is a result, not an infrastructure failure — the
    traceback rides back on stderr so the model can self-correct, matching
    ``execute_tool``'s never-raise contract.
    """
    stdout = stdout.strip()
    stderr = stderr.strip()
    if not stdout and not stderr:
        return (
            "Script ran but produced no output. Print the result you want back "
            "(e.g. print(...))."
        )
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        label = "Error output:" if returncode != 0 else "stderr:"
        parts.append(f"{label}\n{stderr}")
    return _clip("\n\n".join(parts), limit)


def run_python(code: str, settings: Settings | None = None) -> str:
    """Execute ``code`` as a Python 3 script and return its output as a string.

    Never raises: timeouts, missing interpreter, and script exceptions all come
    back as the result text so the tool loop always produces a ``ToolMessage``.
    """
    settings = settings or get_settings()
    limit = settings.code_exec_max_output_chars

    with _exec_slot(settings), tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(_PREAMBLE + "\n" + code, encoding="utf-8")

        # Minimal env: never expose the parent's tokens/secrets. TMPDIR points
        # at the throwaway cwd so stdlib temp writes stay contained.
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "TMPDIR": tmp}

        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=settings.code_exec_timeout,
                preexec_fn=_rlimits(settings),
            )
        except subprocess.TimeoutExpired:
            return (
                f"Script timed out after {settings.code_exec_timeout}s and was "
                "killed. Reduce the work or avoid unbounded loops."
            )
        except FileNotFoundError:
            logger.error("python interpreter %r not found", sys.executable)
            return "Tool failed: the Python interpreter is unavailable."

        return _format_result(
            result.stdout, result.stderr, result.returncode, limit
        )
