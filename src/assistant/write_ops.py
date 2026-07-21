"""Generic target-resolution/undo-logging scaffolding shared by the ops twins.

The calendar and tasks subsystems apply parsed write operations the same way:
resolve the target row by id or a fuzzy title fallback that refuses ambiguity,
mutate the store, then log the applied write to the undo ledger under the
turn's batch. The subsystem supplies a :class:`WriteOpsSpec` naming its finder,
ledger recorder, and an optional pre-write refusal hook; keeping the driver
here means the twins cannot drift apart — the same rationale as
:mod:`assistant.write_ledger`, whose spec this mirrors.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from .config import Settings

logger = logging.getLogger(__name__)


class SupportsIdTitle(Protocol):
    id: str
    title: str


class WriteOpsSpec[RowT: SupportsIdTitle]:
    """How one subsystem's ops resolve write targets and log to its undo ledger."""

    def __init__(
        self,
        kind: str,
        noun: str,
        find: Callable[[Settings, str], list[RowT]],
        title_is_new_value: frozenset[str],
        record_write: Callable[..., None],
        refuse_match: Callable[[Settings, dict, RowT], bool] | None = None,
    ) -> None:
        self.kind = kind  # log prefix: "task" / "calendar"
        self.noun = noun  # log plural: "tasks" / "events"
        self.find = find
        # Ops whose schema defines "title" as the NEW value, not an identifier.
        # Looking the target up by it would resolve to whichever row already
        # bears the new name — one the user never referred to.
        self.title_is_new_value = title_is_new_value
        self.record_write = record_write
        # Veto a resolved single match before it is returned (e.g. the calendar
        # refuses rows mirrored from a read-only ICS feed).
        self.refuse_match = refuse_match


def resolve_target[RowT: SupportsIdTitle](
    spec: WriteOpsSpec[RowT], settings: Settings, op: dict
) -> str | list[RowT] | None:
    """Resolve the row an op refers to, by id or a fuzzy title/query fallback.

    Every op that lands here mutates or deletes, so a fuzzy reference matching
    more than one row is never guessed at — cancelling nothing beats cancelling
    the wrong appointment (the same rule memory's fuzzy forget applies).
    Returns the resolved id on a single unambiguous match; the list of
    candidate rows when the fuzzy reference matches more than one (the caller
    builds the model-facing "which one?" message — this layer only detects the
    collision, it never guesses); or None when nothing matches, or the single
    match was vetoed by refuse_match. An exact id always resolves unambiguously.
    """
    ident = op.get("id") or op.get("query")
    if not ident and op["op"] not in spec.title_is_new_value:
        ident = op.get("title")
    if not ident:
        return None
    matches = spec.find(settings, str(ident))
    if len(matches) > 1:
        logger.warning(
            "%s %s target %r is ambiguous between %d %s (%s); returning candidates",
            spec.kind, op.get("op"), ident, len(matches), spec.noun,
            ", ".join(m.title for m in matches[:5]),
        )
        return matches
    if matches and spec.refuse_match is not None and spec.refuse_match(settings, op, matches[0]):
        return None
    return matches[0].id if matches else None


def log_write[RowT: SupportsIdTitle](
    spec: WriteOpsSpec[RowT],
    settings: Settings,
    thread_id: str,
    batch_id: str,
    target_id: str,
    op: str,
    summary: str,
    before: RowT | None,
) -> None:
    """Record one applied write to the undo ledger, when confirmation is on."""
    if not (thread_id and batch_id and settings.enable_write_confirmation):
        return
    spec.record_write(settings, thread_id, batch_id, target_id, op, summary, before)
