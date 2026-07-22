"""Model-dispatched tools — the assistant's hands.

Split into one module per tool family (mirroring the calendar/ tasks/ docs/
packages); this re-exports the surface the rest of the assistant imports.
See :mod:`.registry` for assembly/dispatch and :mod:`._base` for the shared
ToolContext/ToolSpec records."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec
from .registry import available_tools, execute_tool, tool_map

__all__ = ["ToolContext", "ToolSpec", "available_tools", "execute_tool", "tool_map"]
