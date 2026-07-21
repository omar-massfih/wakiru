"""Tests for the run_python executor — output capture, limits, and isolation.

These run a real Python subprocess (the executor's whole point is that it does),
so they are slower than the pure-code tool tests but exercise the timeout,
truncation, env stripping, and network-block behavior end to end.
"""

from __future__ import annotations

import sys

import pytest

from assistant.code_exec import run_python
from assistant.config import Settings


@pytest.fixture
def settings() -> Settings:
    # Short timeout keeps the timeout test quick; small output cap makes
    # truncation observable without generating megabytes.
    return Settings(
        enable_code_execution=True,
        code_exec_timeout=3,
        code_exec_max_output_chars=200,
    )


def test_stdout_is_returned(settings) -> None:
    assert run_python("print(6 * 7)", settings).strip() == "42"


def test_no_output_nudges_to_print(settings) -> None:
    out = run_python("x = 1 + 1", settings)
    assert "no output" in out.lower()
    assert "print" in out.lower()


def test_exception_comes_back_as_result_not_a_raise(settings) -> None:
    out = run_python("raise ValueError('boom')", settings)
    # No exception propagates; the traceback is handed to the model.
    assert "ValueError" in out
    assert "boom" in out


def test_timeout_is_reported_and_killed(settings) -> None:
    out = run_python("while True:\n    pass", settings)
    assert "timed out" in out.lower()


def test_output_is_truncated(settings) -> None:
    out = run_python("print('x' * 5000)", settings)
    assert "[truncated" in out
    assert len(out) <= settings.code_exec_max_output_chars + 100


def test_environment_secrets_are_not_visible(settings, monkeypatch) -> None:
    # A secret in the parent env must not leak into the sandboxed child.
    monkeypatch.setenv("SOME_TOKEN", "super-secret")
    out = run_python("import os; print(repr(os.environ.get('SOME_TOKEN')))", settings)
    assert "super-secret" not in out
    assert "None" in out


def test_numpy_and_pandas_are_available() -> None:
    # A cold pandas import is heavy, so this one gets the realistic default
    # timeout rather than the deliberately short fixture.
    settings = Settings(enable_code_execution=True, code_exec_timeout=30)
    out = run_python(
        "import numpy as np, pandas as pd\n"
        "print(int(np.array([1, 2, 3]).sum()), len(pd.DataFrame({'a': [1, 2]})))",
        settings,
    )
    assert out.strip() == "6 2"


def test_network_is_blocked(settings) -> None:
    out = run_python(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('example.com', 80), timeout=1)\n"
        "    print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('BLOCKED')\n",
        settings,
    )
    assert "BLOCKED" in out
    assert "CONNECTED" not in out


# RLIMIT_AS is only reliably enforced on Linux (macOS ignores it) — and Linux is
# the deploy target, so that is where the guarantee matters.
@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_AS enforced on Linux"
)
def test_memory_limit_kills_a_greedy_allocation() -> None:
    settings = Settings(
        enable_code_execution=True,
        code_exec_timeout=5,
        code_exec_max_memory_mb=256,
    )
    out = run_python(
        "x = bytearray(4 * 1024 * 1024 * 1024)\nprint('ALLOCATED')", settings
    )
    assert "ALLOCATED" not in out
