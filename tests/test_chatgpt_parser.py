"""Unit tests for the SSE decoding + ChatGptStreamParser state machine.

The end-to-end behavior (HTTP, auth refresh, retries) is covered by the
fake-urlopen tests in test_chatgpt_backend.py; these pin the pure parsing logic
without any I/O.
"""

import json

from assistant.chatgpt_backend import ChatGptStreamParser, iter_sse


def _frame(etype: str, data: dict) -> list[str]:
    return [f"event: {etype}\n", f"data: {json.dumps(data)}\n", "\n"]


# --- iter_sse ---------------------------------------------------------------- #


def test_iter_sse_decodes_event_and_data_frames() -> None:
    lines = _frame("response.output_text.delta", {"delta": "Hi"})
    assert list(iter_sse(lines)) == [("response.output_text.delta", {"delta": "Hi"})]


def test_iter_sse_accepts_bytes_and_crlf() -> None:
    lines = [b"event: response.completed\r\n", b'data: {"ok": true}\r\n', b"\r\n"]
    assert list(iter_sse(lines)) == [("response.completed", {"ok": True})]


def test_iter_sse_joins_multiline_data() -> None:
    payload = json.dumps({"delta": "a\nb"}, indent=0)  # spans two data: lines
    lines = ["event: e\n"] + [f"data: {part}\n" for part in payload.split("\n")] + ["\n"]
    assert list(iter_sse(lines)) == [("e", {"delta": "a\nb"})]


def test_iter_sse_falls_back_to_payload_type_field() -> None:
    lines = ['data: {"type": "response.completed"}\n', "\n"]
    assert list(iter_sse(lines)) == [
        ("response.completed", {"type": "response.completed"})
    ]


def test_iter_sse_skips_done_sentinel_and_non_json() -> None:
    lines = [
        "data: [DONE]\n",
        "\n",
        "data: not json\n",
        "\n",
        'data: [1, 2]\n',  # JSON but not an object
        "\n",
        ": keep-alive comment\n",
        "\n",
    ]
    assert list(iter_sse(lines)) == []


def test_iter_sse_frame_without_trailing_data_is_dropped() -> None:
    # A dangling event: line with no data and no terminator yields nothing.
    assert list(iter_sse(["event: response.completed\n"])) == []


# --- ChatGptStreamParser ------------------------------------------------------ #


def test_deltas_accumulate() -> None:
    parser = ChatGptStreamParser()
    assert parser.feed("response.output_text.delta", {"delta": "Hel"}) == ["Hel"]
    assert parser.feed("response.output_text.delta", {"delta": "lo"}) == ["lo"]
    assert parser.emitted == "Hello"
    assert parser.failure is None and parser.completed is False


def test_completed_sets_flag_without_emitting() -> None:
    parser = ChatGptStreamParser()
    assert parser.feed("response.completed", {"response": {}}) == []
    assert parser.completed is True


def test_failed_collects_error_message() -> None:
    parser = ChatGptStreamParser()
    parser.feed(
        "response.failed",
        {"response": {"error": {"message": "usage limit hit"}}},
    )
    assert parser.failure == "usage limit hit"


def test_error_event_does_not_overwrite_existing_failure() -> None:
    parser = ChatGptStreamParser()
    parser.feed("response.failed", {"response": {"error": {"message": "real"}}})
    parser.feed("error", {"message": "later"})
    assert parser.failure == "real"


def test_reasoning_and_bookkeeping_events_are_ignored() -> None:
    parser = ChatGptStreamParser()
    assert parser.feed("response.reasoning_summary_text.delta", {"delta": "hmm"}) == []
    assert parser.feed("response.output_item.done", {"item": {}}) == []
    assert parser.emitted == ""


def test_empty_or_non_string_delta_is_ignored() -> None:
    parser = ChatGptStreamParser()
    assert parser.feed("response.output_text.delta", {"delta": ""}) == []
    assert parser.feed("response.output_text.delta", {"delta": 42}) == []
    assert parser.emitted == ""
