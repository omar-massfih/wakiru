"""Shared parser for the extractors' JSON op-list replies.

Every background extractor (calendar, tasks, memory learning — and
consolidation, which reuses the memory op set) asks the model for a JSON array
of operation objects. This is the one parser for those replies — tolerant of
code fences and surrounding chatter, strict about shape: anything that isn't a
dict with an allowed ``op`` is dropped.
"""

from __future__ import annotations

import json
import re


def parse_ops(text: str, allowed: frozenset[str]) -> list[dict]:
    """Extract the op dicts from a model reply; unknown/malformed entries drop."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("op") in allowed]
