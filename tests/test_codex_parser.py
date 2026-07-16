"""Unit tests for CodexStreamParser — the JSONL→delta state machine.

The end-to-end stream behavior (subprocess, watchdog, fallback file) is covered
by the fake-codex tests in test_agent.py; these pin the pure parsing logic
without spawning anything.
"""

import json

from assistant.codex_runner import CodexStreamParser


def _msg(etype: str, item_id: str, text: str) -> str:
    return json.dumps(
        {"type": etype, "item": {"id": item_id, "type": "agent_message", "text": text}}
    )


def test_growing_snapshots_reduce_to_increments() -> None:
    parser = CodexStreamParser()
    deltas = []
    for snapshot in ("Hel", "Hello", "Hello world"):
        deltas += parser.feed(_msg("item.updated", "m1", snapshot))
    assert deltas == ["Hel", "lo", " world"]
    assert parser.emitted == "Hello world"
    assert parser.failure is None


def test_item_completed_emits_whole_message() -> None:
    parser = CodexStreamParser()
    assert parser.feed(_msg("item.completed", "m1", "done")) == ["done"]


def test_new_item_id_inserts_separator() -> None:
    parser = CodexStreamParser()
    assert parser.feed(_msg("item.completed", "m1", "first")) == ["first"]
    assert parser.feed(_msg("item.completed", "m2", "second")) == ["\n\n", "second"]


def test_diverged_snapshot_resyncs_whole_text() -> None:
    parser = CodexStreamParser()
    assert parser.feed(_msg("item.updated", "m1", "abc")) == ["abc"]
    assert parser.feed(_msg("item.updated", "m1", "xyz")) == ["\nxyz"]
    assert parser.emitted == "xyz"


def test_unchanged_snapshot_yields_nothing() -> None:
    parser = CodexStreamParser()
    parser.feed(_msg("item.updated", "m1", "same"))
    assert parser.feed(_msg("item.updated", "m1", "same")) == []


def test_non_json_and_foreign_items_are_ignored() -> None:
    parser = CodexStreamParser()
    assert parser.feed("plain log chatter\n") == []
    assert parser.feed(json.dumps({"type": "item.updated", "item": {"type": "reasoning"}})) == []
    assert parser.feed(json.dumps({"type": "turn.started"})) == []
    assert parser.emitted == ""


def test_turn_failed_overwrites_earlier_error() -> None:
    parser = CodexStreamParser()
    parser.feed(json.dumps({"type": "error", "message": "early"}))
    assert parser.failure == "early"
    parser.feed(json.dumps({"type": "turn.failed", "error": {"message": "usage limit hit"}}))
    assert parser.failure == "usage limit hit"


def test_error_does_not_overwrite_existing_failure() -> None:
    parser = CodexStreamParser()
    parser.feed(json.dumps({"type": "turn.failed", "error": {"message": "boom"}}))
    parser.feed(json.dumps({"type": "error", "message": "later"}))
    assert parser.failure == "boom"


def test_empty_turn_failed_keeps_prior_failure() -> None:
    parser = CodexStreamParser()
    parser.feed(json.dumps({"type": "turn.failed", "error": {"message": "real"}}))
    parser.feed(json.dumps({"type": "turn.failed", "error": {}}))
    assert parser.failure == "real"
