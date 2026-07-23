"""People writes — the operation set the agent's people tools apply.

Each operation arrives as a parsed dict —

* ``add``          — record a new person,
* ``update``       — change an existing person's fields,
* ``log_contact``  — stamp that the user was just in touch with them,
* ``remove``       — delete a person.

Existing people are targeted by id or a fuzzy name reference that refuses
ambiguity, exactly as :mod:`assistant.tasks.ops` does. Every applied op is
logged to the undo ledger under the turn's batch, so "undo" reverts it.
"""

from __future__ import annotations

from .. import write_ops
from ..config import Settings
from . import store, undo

# The find/record_write lambdas resolve attributes at call time so test
# monkeypatches on those modules keep working (see tasks.ops for the rationale).
_SPEC: write_ops.WriteOpsSpec[store.Person] = write_ops.WriteOpsSpec(
    kind="person",
    noun="people",
    find=lambda settings, ident: store.find_people(settings, ident),
    title_is_new_value=frozenset(),  # people resolve by id/query, never by name-as-target
    record_write=lambda *args: undo.record_write(*args),
)


def _target_id(settings: Settings, op: dict) -> str | list[store.Person] | None:
    return write_ops.resolve_target(_SPEC, settings, op)


def _describe(p: store.Person) -> str:
    rel = f", {p.relationship}" if p.relationship else ""
    return f'{p.id} ("{p.name}"{rel})'


def _dedupe_message(existing: store.Person) -> str:
    return (
        f"Not added — a person with this exact name already exists: "
        f"{_describe(existing)}. Use update_person with id {existing.id} to "
        "change them, or add_person with a more distinguishing name if this is "
        "genuinely someone else."
    )


def _ambiguous_message(matches: list[store.Person]) -> str:
    shown = ", ".join(_describe(m) for m in matches[:5])
    more = f", +{len(matches) - 5} more" if len(matches) > 5 else ""
    return (
        f"Ambiguous — {len(matches)} people match: {shown}{more}. "
        "Retry with one exact id from the People block."
    )


def _log_write(
    settings: Settings,
    thread_id: str,
    batch_id: str,
    person_id: str,
    op: str,
    summary: str,
    before: store.Person | None,
) -> None:
    write_ops.log_write(_SPEC, settings, thread_id, batch_id, person_id, op, summary, before)


def apply_op(
    settings: Settings, op: dict, thread_id: str = "", batch_id: str = ""
) -> str | None:
    """Apply a single parsed operation; return a short log line, a
    dedupe/ambiguous-match message for the model to act on, or ``None``
    (nothing found/nothing to do)."""
    kind = op["op"]

    if kind == "add" and op.get("name"):
        dupe = store.find_exact_name(settings, str(op["name"]))
        if dupe is not None:
            return _dedupe_message(dupe)
        person = store.create_person(
            settings,
            name=str(op["name"]),
            relationship=str(op.get("relationship", "") or ""),
            cadence_days=op.get("cadence_days", 0) or 0,
            birthday=str(op.get("birthday", "") or ""),
            notes=str(op.get("notes", "") or ""),
        )
        summary = f"added person: {person.name}"
        _log_write(settings, thread_id, batch_id, person.id, "add", summary, None)
        return summary

    if kind == "update":
        target = _target_id(settings, op)
        if isinstance(target, list):
            return _ambiguous_message(target)
        if target is None:
            return None
        before = store.get_person(settings, target)
        revised = store.update_person(
            settings, target,
            name=op.get("name"), relationship=op.get("relationship"),
            cadence_days=op.get("cadence_days"), birthday=op.get("birthday"),
            notes=op.get("notes"),
        )
        if revised is None:
            return None
        summary = f"updated person: {revised.name}"
        _log_write(settings, thread_id, batch_id, target, "update", summary, before)
        return summary

    if kind == "log_contact":
        target = _target_id(settings, op)
        if isinstance(target, list):
            return _ambiguous_message(target)
        if target is None:
            return None
        before = store.get_person(settings, target)
        contacted = store.log_contact(settings, target, when=str(op.get("when", "") or ""))
        if contacted is None:
            return None
        summary = f"logged contact with {contacted.name}"
        _log_write(settings, thread_id, batch_id, target, "log_contact", summary, before)
        return summary

    if kind == "remove":
        target = _target_id(settings, op)
        if isinstance(target, list):
            return _ambiguous_message(target)
        if target is None:
            return None
        deleted = store.delete_person(settings, target)
        if deleted is None:
            return None
        summary = f"removed person: {deleted.name}"
        _log_write(settings, thread_id, batch_id, target, "remove", summary, deleted)
        return summary

    return None
