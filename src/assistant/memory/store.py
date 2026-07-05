"""File-backed memory store: markdown notes with YAML frontmatter.

Each long-term memory is one ``.md`` file — the source of truth — laid out as::

    memory/
      MEMORY.md          # regenerated index (one line per note)
      facts/<name>.md    # type == "fact"
      learnings/<name>.md# type == "learning"

The files are plain markdown so a human (or Codex, in a widened sandbox) can read
and edit them directly. The vector index in :mod:`.index` is derived from these
files, never the other way around.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from ..config import Settings

# type -> subdirectory
_CATEGORY = {"fact": "facts", "learning": "learnings"}
_DEFAULT_TYPE = "fact"

INDEX_FILENAME = "MEMORY.md"


@dataclass
class Note:
    """A single long-term memory."""

    name: str
    description: str
    body: str
    type: str = _DEFAULT_TYPE
    created: str = field(default_factory=lambda: date.today().isoformat())
    updated: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def category(self) -> str:
        return _CATEGORY.get(self.type, _CATEGORY[_DEFAULT_TYPE])

    @property
    def index_text(self) -> str:
        """Text fed to the embedder — description plus body for good recall."""
        return f"{self.description}\n\n{self.body}".strip()

    def render(self) -> str:
        """Serialize to a frontmatter markdown document."""
        meta = {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "created": self.created,
            "updated": self.updated,
        }
        front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{front}\n---\n\n{self.body.strip()}\n"


def slugify(text: str, max_words: int = 6) -> str:
    """Turn free text into a short kebab-case slug suitable for a filename."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(words[:max_words]) or "memory"
    return slug[:60].strip("-")


def memory_root(settings: Settings) -> Path:
    root = settings.memory_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def note_path(settings: Settings, note: Note) -> Path:
    return memory_root(settings) / note.category / f"{note.name}.md"


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def parse_note(text: str) -> Note:
    """Parse a frontmatter markdown document back into a :class:`Note`."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("note is missing frontmatter")
    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    return Note(
        name=meta["name"],
        description=meta.get("description", ""),
        body=body,
        type=meta.get("type", _DEFAULT_TYPE),
        created=str(meta.get("created", date.today().isoformat())),
        updated=str(meta.get("updated", date.today().isoformat())),
    )


def read_note(path: Path) -> Note:
    return parse_note(path.read_text(encoding="utf-8"))


def write_note(settings: Settings, note: Note) -> Path:
    """Write a note to disk, creating its category directory as needed."""
    path = note_path(settings, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note.render(), encoding="utf-8")
    return path


def list_notes(settings: Settings) -> list[Note]:
    """All notes on disk, sorted by name."""
    root = memory_root(settings)
    notes: list[Note] = []
    for path in root.rglob("*.md"):
        if path.name == INDEX_FILENAME:
            continue
        try:
            notes.append(read_note(path))
        except (ValueError, KeyError):
            continue  # skip malformed files rather than crash recall
    return sorted(notes, key=lambda n: n.name)


def find_note(settings: Settings, name: str) -> Note | None:
    for note in list_notes(settings):
        if note.name == name:
            return note
    return None


def delete_note(settings: Settings, name: str) -> Note | None:
    """Delete a note by name; return it if it existed."""
    note = find_note(settings, name)
    if note is None:
        return None
    note_path(settings, note).unlink(missing_ok=True)
    return note


def regenerate_index(settings: Settings) -> Path:
    """Rewrite ``MEMORY.md`` from the notes currently on disk."""
    notes = list_notes(settings)
    lines = ["# Memory index", ""]
    if not notes:
        lines.append("_(empty)_")
    else:
        for note in notes:
            lines.append(f"- **{note.name}** ({note.type}) — {note.description}")
    path = memory_root(settings) / INDEX_FILENAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_index(settings: Settings) -> str:
    """Current ``MEMORY.md`` contents, regenerating if absent."""
    path = memory_root(settings) / INDEX_FILENAME
    if not path.exists():
        regenerate_index(settings)
    return path.read_text(encoding="utf-8").strip()
