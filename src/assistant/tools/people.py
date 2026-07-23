"""People tools — add/update/log-contact/remove over the people store."""
from __future__ import annotations

from ._base import _ISO, _NO_MATCH, ToolContext, ToolSpec, _op_runner, _params


def _person_op(ctx: ToolContext, op: dict) -> str:
    from ..people import ops as people_ops

    result = people_ops.apply_op(ctx.settings, op, ctx.thread_id, ctx.batch_id)
    return result or _NO_MATCH


def _people_tools() -> list[ToolSpec]:
    _ref = "Exact person id from the People block, or their name"
    return [
        ToolSpec(
            "add_person",
            "Record a person the user knows — a friend, family member, "
            "colleague, or contact worth remembering. Capture their "
            "relationship and anything durable the user mentions.",
            _params(
                {
                    "name": ("string", "The person's name"),
                    "relationship": (
                        "string",
                        "How the user knows them (e.g. \"sister\", "
                        "\"colleague at Acme\", \"dentist\")",
                    ),
                    "cadence_days": (
                        "string",
                        "Keep-in-touch interval in days (e.g. \"14\"); omit if "
                        "the user doesn't want reminding to stay in touch",
                    ),
                    "birthday": ("string", "Birthday as MM-DD or YYYY-MM-DD"),
                    "notes": ("string", "Free-form notes about them"),
                },
                ["name"],
            ),
            _op_runner(_person_op, "add"),
        ),
        ToolSpec(
            "update_person",
            "Change a person's name, relationship, keep-in-touch cadence, "
            "birthday, or notes.",
            _params(
                {
                    "query": ("string", _ref),
                    "name": ("string", "New name"),
                    "relationship": ("string", "New relationship"),
                    "cadence_days": ("string", "New keep-in-touch interval in days"),
                    "birthday": ("string", "New birthday, MM-DD or YYYY-MM-DD"),
                    "notes": ("string", "New notes"),
                },
                ["query"],
            ),
            _op_runner(_person_op, "update"),
        ),
        ToolSpec(
            "log_contact",
            "Record that the user was just in touch with someone (resets the "
            "keep-in-touch clock). Use it when the user says they spoke to, met, "
            "called, or messaged a person you track.",
            _params(
                {
                    "query": ("string", _ref),
                    "when": (
                        "string",
                        f"When they were in touch, {_ISO}; omit for now",
                    ),
                },
                ["query"],
            ),
            _op_runner(_person_op, "log_contact"),
        ),
        ToolSpec(
            "remove_person",
            "Delete a person from the store.",
            _params({"query": ("string", _ref)}, ["query"]),
            _op_runner(_person_op, "remove"),
        ),
    ]
