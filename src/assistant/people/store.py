"""SQLite-backed store for the people the user knows (a lightweight CRM).

A single ``people`` table in its own SQLite file (:attr:`Settings.people_db_path`,
under the memory directory), modeled on :mod:`assistant.tasks.store`. A person
carries a ``relationship`` (how the user knows them), an optional keep-in-touch
``cadence_days``, the ``last_contact`` instant, an optional ``birthday``, and
free-form ``notes``. ``last_contact`` — when set — is a timezone-aware ISO-8601
string, stored so the offset travels with the value, exactly as the calendar and
tasks stores do for their times.

A fresh connection is opened per operation with WAL + a busy timeout, so the
store is safe to touch from FastAPI request handlers and background tasks alike.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings, postgres_backend
from ..sqlite_util import open_db, transaction

# Text columns a caller may set on create/update (id + timestamps + cadence
# handled separately: cadence is an int, last_contact has its own log_contact path).
_TEXT_FIELDS = ("name", "relationship", "birthday", "notes")


@dataclass
class Person:
    """One person in the user's circle.

    ``cadence_days`` is the keep-in-touch interval (0 = no cadence tracked).
    ``last_contact`` is an optional tz-aware ISO-8601 string (empty = never
    logged). ``birthday`` is ``MM-DD`` or ``YYYY-MM-DD`` (empty = unknown).
    """

    id: str
    name: str
    relationship: str = ""
    cadence_days: int = 0
    last_contact: str = ""
    birthday: str = ""
    notes: str = ""
    created: str = ""
    updated: str = ""

    @property
    def title(self) -> str:
        """The person's display handle — the name.

        Named ``title`` so :class:`assistant.write_ops.WriteOpsSpec` (which
        resolves and logs write targets generically for tasks, calendar, and
        people alike) can treat a person like any other row.
        """
        return self.name


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.people_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS people ("
        " id TEXT PRIMARY KEY, name TEXT NOT NULL, relationship TEXT DEFAULT '',"
        " cadence_days INTEGER DEFAULT 0, last_contact TEXT DEFAULT '',"
        " birthday TEXT DEFAULT '', notes TEXT DEFAULT '',"
        " created TEXT DEFAULT '', updated TEXT DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, closed on exit (see tasks.store)."""
    with transaction(_open(settings)) as conn:
        yield conn


def _row_to_person(row: sqlite3.Row) -> Person:
    return Person(
        id=row["id"],
        name=row["name"],
        relationship=row["relationship"] or "",
        cadence_days=int(row["cadence_days"] or 0),
        last_contact=row["last_contact"] or "",
        birthday=row["birthday"] or "",
        notes=row["notes"] or "",
        created=row["created"] or "",
        updated=row["updated"] or "",
    )


def _stamp_now(settings: Settings) -> str:
    from ..calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def _normalize_stamp(settings: Settings, value: str) -> str:
    """Attach the assistant's timezone to a naive ISO datetime on its way in.

    Blank or unparseable values pass through unchanged. Mirrors
    tasks.store._normalize_due.
    """
    value = value.strip()
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is not None:
        return value
    from ..calendar.context import resolve_tz

    return dt.replace(tzinfo=resolve_tz(settings)).isoformat()


def _coerce_cadence(value: object, default: int = 0) -> int:
    """A cadence arg (string from a tool, or int) as a non-negative int."""
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(0, n)


def create_person(
    settings: Settings,
    name: str,
    relationship: str = "",
    cadence_days: object = 0,
    birthday: str = "",
    notes: str = "",
) -> Person:
    """Insert a new person and return it (with a generated id and timestamps)."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.create_person(
            settings, name, relationship, _coerce_cadence(cadence_days), birthday, notes
        )
    now = _stamp_now(settings)
    person = Person(
        id=uuid.uuid4().hex[:12],
        name=name.strip(),
        relationship=relationship.strip(),
        cadence_days=_coerce_cadence(cadence_days),
        birthday=birthday.strip(),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO people"
            " (id, name, relationship, cadence_days, last_contact, birthday, notes, created, updated)"
            " VALUES (?, ?, ?, ?, '', ?, ?, ?, ?)",
            (person.id, person.name, person.relationship, person.cadence_days,
             person.birthday, person.notes, person.created, person.updated),
        )
    return person


def get_person(settings: Settings, person_id: str) -> Person | None:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.get_person(settings, person_id)
    with _connect(settings) as conn:
        row = conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
    return _row_to_person(row) if row else None


def list_people(settings: Settings) -> list[Person]:
    """Everyone, ordered by name (case-insensitive)."""
    if storage_postgres := postgres_backend(settings):
        people = storage_postgres.list_people(settings)
    else:
        with _connect(settings) as conn:
            rows = conn.execute("SELECT * FROM people").fetchall()
        people = [_row_to_person(r) for r in rows]
    return sorted(people, key=lambda p: p.name.lower())


def update_person(settings: Settings, person_id: str, **fields: object) -> Person | None:
    """Update the given columns on a person; return it, or ``None`` if absent."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.update_person(settings, person_id, fields)
    existing = get_person(settings, person_id)
    if existing is None:
        return None
    updates: dict[str, object] = {
        k: str(v).strip()
        for k, v in fields.items()
        if k in _TEXT_FIELDS and v is not None
    }
    if fields.get("cadence_days") is not None:
        updates["cadence_days"] = _coerce_cadence(fields["cadence_days"])
    if fields.get("last_contact") is not None:
        updates["last_contact"] = _normalize_stamp(settings, str(fields["last_contact"]))
    if not updates:
        return existing
    updates["updated"] = _stamp_now(settings)
    columns = ", ".join(f"{k} = ?" for k in updates)
    with _connect(settings) as conn:
        conn.execute(
            f"UPDATE people SET {columns} WHERE id = ?",
            (*updates.values(), person_id),
        )
    return get_person(settings, person_id)


def log_contact(settings: Settings, person_id: str, when: str = "") -> Person | None:
    """Stamp ``last_contact`` (defaults to now); return the person, or ``None``."""
    stamp = _normalize_stamp(settings, when) if when.strip() else _stamp_now(settings)
    return update_person(settings, person_id, last_contact=stamp)


def restore_person(settings: Settings, person: Person) -> Person:
    """Re-insert a full person snapshot verbatim, overwriting any current row with
    the same id. Used by the undo path; never bumps ``updated``."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.restore_person(settings, person)
    with _connect(settings) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO people"
            " (id, name, relationship, cadence_days, last_contact, birthday, notes, created, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (person.id, person.name, person.relationship, person.cadence_days,
             person.last_contact, person.birthday, person.notes,
             person.created, person.updated),
        )
    return person


def delete_person(settings: Settings, person_id: str) -> Person | None:
    """Delete a person by id; return them if they existed."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.delete_person(settings, person_id)
    existing = get_person(settings, person_id)
    if existing is None:
        return None
    with _connect(settings) as conn:
        conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    return existing


def find_people(settings: Settings, query: str) -> list[Person]:
    """All candidate people for ``query``: an exact-id match alone, else every
    case-insensitive name-substring match (mirrors tasks.find_tasks)."""
    query = query.strip()
    if not query:
        return []
    exact = get_person(settings, query)
    if exact is not None:
        return [exact]
    needle = query.lower()
    return [p for p in list_people(settings) if needle in p.name.lower()]


def find_person(settings: Settings, query: str) -> Person | None:
    """Resolve ``query`` to a single person: by exact id, else the best name match."""
    matches = find_people(settings, query)
    return matches[0] if matches else None


def find_exact_name(settings: Settings, name: str) -> Person | None:
    """The person whose name exactly matches ``name`` (case-insensitive, stripped),
    or None — used solely to dedupe add_person."""
    needle = name.strip().lower()
    if not needle:
        return None
    for p in list_people(settings):
        if p.name.strip().lower() == needle:
            return p
    return None
