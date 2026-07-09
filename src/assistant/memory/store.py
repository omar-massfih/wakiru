"""File-backed memory store: markdown notes with YAML frontmatter.

Each long-term memory is one ``.md`` file — the source of truth — laid out by
*kind*::

    memory/
      MEMORY.md              # regenerated index (grouped by kind)
      episodic/<name>.md     # kind == "episodic"   — what happened (decays)
      semantic/<name>.md     # kind == "semantic"   — durable facts/preferences
      procedural/<name>.md   # kind == "procedural" — learnings / how-to

The files are plain markdown so a human (or Codex, in a widened sandbox) can read
and edit them directly. The vector index in :mod:`.index` is derived from these
files, never the other way around — :func:`.index.reindex` rebuilds it from disk.

Frontmatter carries the signals the brain learns from: ``salience`` and
``confidence`` (importance / trust), timestamps, and the *soft* reinforcement
counters ``recall_count`` / ``last_recalled``. The counters are authoritative in
the index DB and mirrored back here on consolidation, so a hand-edit never has to
touch them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from ..config import Settings

# The three cognitive stores; each kind lives in a same-named subdirectory.
_KINDS = frozenset({"episodic", "semantic", "procedural"})
_DEFAULT_KIND = "semantic"

# Back-compat: the old two-value ``type`` and its directories.
_LEGACY_KIND = {"fact": "semantic", "learning": "procedural"}

INDEX_FILENAME = "MEMORY.md"

# Deleted notes are moved here (under the memory root) instead of unlinked, so
# an LLM-driven forget is recoverable by hand until consolidation prunes it.
TRASH_DIRNAME = ".trash"
_TRASH_STAMP_FORMAT = "%Y%m%dT%H%M%S"

# Human-friendly order for the grouped MEMORY.md listing.
_KIND_ORDER = ["semantic", "procedural", "episodic"]


def _today() -> str:
    return date.today().isoformat()


def normalize_kind(kind: str | None) -> str | None:
    """Map legacy kind names ("fact"/"learning") to current ones; ``None`` if unknown.

    The single gate for kind values arriving from outside (LLM ops, old
    frontmatter) — an unrecognized kind must never reach the index, where it
    would dodge the per-kind caps, biases, and dedup matching.
    """
    if not kind:
        return None
    kind = _LEGACY_KIND.get(kind, kind)
    return kind if kind in _KINDS else None


@dataclass
class Note:
    """A single long-term memory."""

    name: str
    description: str
    body: str
    kind: str = _DEFAULT_KIND
    salience: float = 0.5
    confidence: float = 0.8
    tags: list[str] = field(default_factory=list)
    source: str = ""
    created: str = field(default_factory=_today)
    updated: str = field(default_factory=_today)
    last_recalled: str = ""
    recall_count: int = 0

    def __post_init__(self) -> None:
        self.kind = normalize_kind(self.kind) or _DEFAULT_KIND

    @property
    def index_text(self) -> str:
        """Text fed to the embedder — description plus body for good recall."""
        return f"{self.description}\n\n{self.body}".strip()

    def render(self) -> str:
        """Serialize to a frontmatter markdown document."""
        meta = {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "salience": round(float(self.salience), 3),
            "confidence": round(float(self.confidence), 3),
            "tags": list(self.tags),
            "source": self.source,
            "created": self.created,
            "updated": self.updated,
            "last_recalled": self.last_recalled,
            "recall_count": int(self.recall_count),
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
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.virtual_note_path(settings, note)
    # note.kind is always a valid kind (normalized in __post_init__).
    return memory_root(settings) / note.kind / f"{note.name}.md"


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_note(text: str) -> Note:
    """Parse a frontmatter markdown document back into a :class:`Note`."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("note is missing frontmatter")
    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    # Accept either the new ``kind`` or the legacy ``type`` field.
    kind = meta.get("kind", meta.get("type", _DEFAULT_KIND))
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return Note(
        name=meta["name"],
        description=meta.get("description", ""),
        body=body,
        kind=str(kind),
        salience=_as_float(meta.get("salience"), 0.5),
        confidence=_as_float(meta.get("confidence"), 0.8),
        tags=list(tags),
        source=str(meta.get("source", "")),
        created=str(meta.get("created", _today())),
        updated=str(meta.get("updated", _today())),
        last_recalled=str(meta.get("last_recalled", "") or ""),
        recall_count=int(_as_float(meta.get("recall_count"), 0)),
    )


def read_note(path: Path) -> Note:
    return parse_note(path.read_text(encoding="utf-8"))


def unique_name(settings: Settings, slug: str, keep: str | None = None) -> str:
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.unique_name(settings, slug, keep)
    """A note name that won't clobber a *different* existing note.

    If ``slug`` is free, or already belongs to ``keep`` (the note we intend to
    overwrite/update), return it as-is. Otherwise append ``-2``, ``-3``… until a
    free name is found. This prevents two unrelated facts whose descriptions
    slugify identically from silently overwriting each other.
    """
    existing = {n.name for n in list_notes(settings)}
    if slug == keep or slug not in existing:
        return slug
    i = 2
    while f"{slug}-{i}" in existing:
        i += 1
    return f"{slug}-{i}"


def write_note(settings: Settings, note: Note) -> Path:
    """Write a note to the configured store."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.write_note(settings, note)
    path = note_path(settings, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note.render(), encoding="utf-8")
    return path


def list_notes(settings: Settings) -> list[Note]:
    """All live notes, sorted by name (trashed notes excluded)."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.list_notes(settings)
    root = memory_root(settings)
    notes: list[Note] = []
    for path in root.rglob("*.md"):
        if path.name == INDEX_FILENAME or TRASH_DIRNAME in path.parts:
            continue
        try:
            notes.append(read_note(path))
        except (ValueError, KeyError):
            continue  # skip malformed files rather than crash recall
    return sorted(notes, key=lambda n: n.name)


def find_note(settings: Settings, name: str) -> Note | None:
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.find_note(settings, name)
    """Look up a note by name without scanning the whole store.

    A note's path is deterministic per kind, so probe the three candidate
    paths directly — this runs several times per save, and a full ``rglob``
    scan grows linearly with the store.
    """
    root = memory_root(settings)
    for kind_dir in _KINDS:
        path = root / kind_dir / f"{name}.md"
        if path.exists():
            try:
                return read_note(path)
            except (ValueError, KeyError):
                continue  # malformed file — same skip rule as list_notes
    return None


def purge_stale_files(settings: Settings, name: str, keep_kind: str) -> None:
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        storage_postgres.purge_stale_files(settings, name, keep_kind)
        return
    """Delete any ``<name>.md`` living under a kind dir other than ``keep_kind``.

    Guarantees one file per note name even when a note changes kind (e.g. an
    episode promoted to semantic), so a rename across directories never leaves a
    stale duplicate behind.
    """
    root = memory_root(settings)
    keep_dir = keep_kind if keep_kind in _KINDS else _DEFAULT_KIND
    for kind_dir in _KINDS:
        if kind_dir == keep_dir:
            continue
        (root / kind_dir / f"{name}.md").unlink(missing_ok=True)


def delete_note(settings: Settings, name: str) -> Note | None:
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.delete_note(settings, name)
    """Soft-delete a note by name; return it if it existed.

    The file is moved into ``.trash/`` (timestamp-prefixed) rather than
    unlinked, so a deletion decided by an LLM op — which happens with no user
    confirmation — stays recoverable by hand until :func:`prune_trash` ages it
    out during consolidation.
    """
    note = find_note(settings, name)
    if note is None:
        return None
    path = note_path(settings, note)
    trash = memory_root(settings) / TRASH_DIRNAME
    trash.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime(_TRASH_STAMP_FORMAT)
    target = trash / f"{stamp}-{note.kind}-{note.name}.md"
    i = 2
    while target.exists():
        target = trash / f"{stamp}-{note.kind}-{note.name}-{i}.md"
        i += 1
    try:
        path.replace(target)
    except FileNotFoundError:
        pass  # already gone — the deletion still "happened"
    return note


def prune_trash(settings: Settings, max_age_days: int) -> int:
    """Permanently remove trashed notes older than ``max_age_days``."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.prune_trash(settings, max_age_days)
    trash = memory_root(settings) / TRASH_DIRNAME
    if not trash.exists():
        return 0
    now = datetime.now().astimezone()
    pruned = 0
    for path in trash.glob("*.md"):
        try:
            deleted_at = datetime.strptime(
                path.name.split("-", 1)[0], _TRASH_STAMP_FORMAT
            ).astimezone()
        except ValueError:
            # No parseable stamp — fall back to the file's mtime.
            deleted_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        if (now - deleted_at).days >= max_age_days:
            path.unlink(missing_ok=True)
            pruned += 1
    return pruned


def regenerate_index(settings: Settings) -> Path:
    """Rewrite/export ``MEMORY.md`` from the configured note store."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.regenerate_index(settings)
    notes = list_notes(settings)
    lines = ["# Memory index", ""]
    if not notes:
        lines.append("_(empty)_")
    else:
        by_kind: dict[str, list[Note]] = {}
        for note in notes:
            by_kind.setdefault(note.kind, []).append(note)
        order = _KIND_ORDER + [k for k in by_kind if k not in _KIND_ORDER]
        for kind in order:
            group = by_kind.get(kind)
            if not group:
                continue
            lines.append(f"## {kind.capitalize()}")
            for note in group:
                lines.append(f"- **{note.name}** — {note.description}")
            lines.append("")
    path = memory_root(settings) / INDEX_FILENAME
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def read_index(settings: Settings) -> str:
    """Current ``MEMORY.md`` contents, regenerating if absent."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.read_index(settings)
    path = memory_root(settings) / INDEX_FILENAME
    if not path.exists():
        regenerate_index(settings)
    return path.read_text(encoding="utf-8").strip()
