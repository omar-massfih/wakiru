"""People tables for the Postgres backend (twin of :mod:`assistant.people.store`)."""

from __future__ import annotations

from ..config import Settings
from .core import (
    _rows,
    _schema_done,
    _schema_mark,
    connect,
)


def ensure_people_schema(settings: Settings) -> None:
    if _schema_done(settings, "people"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_people (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              relationship TEXT NOT NULL DEFAULT '',
              cadence_days INTEGER NOT NULL DEFAULT 0,
              last_contact TEXT NOT NULL DEFAULT '',
              birthday TEXT NOT NULL DEFAULT '',
              notes TEXT NOT NULL DEFAULT '',
              created TEXT NOT NULL DEFAULT '',
              updated TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_person_write_log (
              id BIGSERIAL PRIMARY KEY,
              thread_id TEXT NOT NULL,
              batch_id TEXT NOT NULL,
              person_id TEXT NOT NULL,
              op TEXT NOT NULL,
              summary TEXT NOT NULL,
              before_json TEXT,
              applied_at TEXT NOT NULL,
              undone_at TEXT
            )
            """
        )
    _schema_mark(settings, "people")


def _person_from_row(row: dict):
    from ..people.store import Person

    return Person(
        id=str(row["id"]),
        name=str(row["name"]),
        relationship=str(row.get("relationship") or ""),
        cadence_days=int(row.get("cadence_days") or 0),
        last_contact=str(row.get("last_contact") or ""),
        birthday=str(row.get("birthday") or ""),
        notes=str(row.get("notes") or ""),
        created=str(row.get("created") or ""),
        updated=str(row.get("updated") or ""),
    )


_COLS = "id, name, relationship, cadence_days, last_contact, birthday, notes, created, updated"


def create_person(
    settings: Settings,
    name: str,
    relationship: str = "",
    cadence_days: int = 0,
    birthday: str = "",
    notes: str = "",
):
    import uuid

    from ..people import store as people_store

    ensure_people_schema(settings)
    now = people_store._stamp_now(settings)
    person = people_store.Person(
        id=uuid.uuid4().hex[:12],
        name=name.strip(),
        relationship=relationship.strip(),
        cadence_days=people_store._coerce_cadence(cadence_days),
        birthday=birthday.strip(),
        notes=notes.strip(),
        created=now,
        updated=now,
    )
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_people"
            " (id, name, relationship, cadence_days, last_contact, birthday, notes, created, updated)"
            " VALUES (%s, %s, %s, %s, '', %s, %s, %s, %s)",
            (person.id, person.name, person.relationship, person.cadence_days,
             person.birthday, person.notes, person.created, person.updated),
        )
    return person


def get_person(settings: Settings, person_id: str):
    ensure_people_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(
                f"SELECT {_COLS} FROM assistant_people WHERE id = %s", (person_id,)
            )
        )
    return _person_from_row(rows[0]) if rows else None


def list_people(settings: Settings):
    ensure_people_schema(settings)
    with connect(settings) as conn:
        rows = _rows(conn.execute(f"SELECT {_COLS} FROM assistant_people"))
    return [_person_from_row(r) for r in rows]


def update_person(settings: Settings, person_id: str, fields: dict):
    from ..people import store as people_store

    ensure_people_schema(settings)
    existing = get_person(settings, person_id)
    if existing is None:
        return None
    updates: dict[str, object] = {
        k: str(v).strip()
        for k, v in fields.items()
        if k in people_store._TEXT_FIELDS and v is not None
    }
    if fields.get("cadence_days") is not None:
        updates["cadence_days"] = people_store._coerce_cadence(fields["cadence_days"])
    if fields.get("last_contact") is not None:
        updates["last_contact"] = people_store._normalize_stamp(
            settings, str(fields["last_contact"])
        )
    if not updates:
        return existing
    updates["updated"] = people_store._stamp_now(settings)
    assignments = ", ".join(f"{k} = %s" for k in updates)
    with connect(settings) as conn:
        conn.execute(
            f"UPDATE assistant_people SET {assignments} WHERE id = %s",
            (*updates.values(), person_id),
        )
    return get_person(settings, person_id)


def restore_person(settings: Settings, person) -> object:
    ensure_people_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO assistant_people
              (id, name, relationship, cadence_days, last_contact, birthday, notes, created, updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id) DO UPDATE SET
              name = excluded.name,
              relationship = excluded.relationship,
              cadence_days = excluded.cadence_days,
              last_contact = excluded.last_contact,
              birthday = excluded.birthday,
              notes = excluded.notes,
              created = excluded.created,
              updated = excluded.updated
            """,
            (person.id, person.name, person.relationship, person.cadence_days,
             person.last_contact, person.birthday, person.notes,
             person.created, person.updated),
        )
    return person


def delete_person(settings: Settings, person_id: str):
    existing = get_person(settings, person_id)
    if existing is None:
        return None
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_people WHERE id = %s", (person_id,))
    return existing
