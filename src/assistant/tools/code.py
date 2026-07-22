"""Code-execution tool — run a short Python script over data in hand."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _params


def _run_python(ctx: ToolContext, code: str) -> str:
    from .. import code_exec

    return code_exec.run_python(str(code), ctx.settings)

def _code_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "run_python",
            "Run a short Python 3 script to compute over data you already have "
            "(from documents, attachments, email, or the calendar). The "
            "standard library plus numpy and pandas are available; there is no "
            "network and no access to the user's files or stores — pull "
            "anything you need with the other tools first and pass it inline in "
            "the code. Return results by printing them.",
            _params(
                {"code": ("string", "A complete Python 3 script; print the result")},
                ["code"],
            ),
            _run_python,
        )
    ]
