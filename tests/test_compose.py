"""Composer tests — background pushes in the assistant's voice, fallback-proof.

The model is faked; the persona and context assembly run for real (with the
same fake embedder the other memory-touching tests use).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from assistant import compose
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), timezone="Europe/Oslo")


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.memory.embeddings._embed",
        lambda texts, prefix="", settings=None: [[1.0] + [0.0] * 63 for _ in texts],
    )


class _CannedModel:
    def __init__(self, reply) -> None:
        self.reply = reply
        self.prompts: list[list] = []

    def invoke(self, messages):
        self.prompts.append(list(messages))
        return AIMessage(content=self.reply)


def _wire(monkeypatch, model) -> None:
    monkeypatch.setattr(compose, "build_model", lambda s=None: model)


def test_composes_with_persona_context_instruction_and_facts(settings, monkeypatch) -> None:
    model = _CannedModel("Snart: tannlegen kl 14.")
    _wire(monkeypatch, model)
    text = compose.compose_push(
        settings,
        instruction="Compose one short nudge.",
        facts="- Dentist at 14:00",
        query="Dentist",
        fallback="Dentist at 14:00 (in 1 hour).",
    )
    assert text == "Snart: tannlegen kl 14."
    joined = "\n".join(str(m.content) for m in model.prompts[0])
    assert "You are Wakiru" in joined  # persona leads
    assert "Current date and time" in joined  # context providers ran
    assert "Compose one short nudge." in joined
    assert "- Dentist at 14:00" in joined


def test_model_failure_returns_the_fallback(settings, monkeypatch) -> None:
    class _Boom:
        def invoke(self, messages):
            raise RuntimeError("model down")

    _wire(monkeypatch, _Boom())
    text = compose.compose_push(
        settings,
        instruction="i",
        facts="f",
        query="q",
        fallback="Dentist at 14:00 (in 1 hour).",
    )
    assert text == "Dentist at 14:00 (in 1 hour)."


def test_empty_reply_returns_the_fallback(settings, monkeypatch) -> None:
    _wire(monkeypatch, _CannedModel("   \n"))
    text = compose.compose_push(
        settings, instruction="i", facts="f", query="q", fallback="fallback text"
    )
    assert text == "fallback text"


def test_anthropic_block_content_is_flattened(settings, monkeypatch) -> None:
    blocks = [{"type": "text", "text": "Hei "}, {"type": "text", "text": "der!"}]
    _wire(monkeypatch, _CannedModel(blocks))
    text = compose.compose_push(
        settings, instruction="i", facts="f", query="q", fallback="nope"
    )
    assert text == "Hei der!"
