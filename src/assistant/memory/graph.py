"""Knowledge graph over the memory notes, stored in SQLite (edge + node tables).

This is the relationship layer the flat note store lacks. Each note may carry
entity-relationship triples in its frontmatter (``Note.relations``); this module
mirrors them into two tables so recall can *traverse* them — connecting
``Omar -sister-> Sara`` with ``Sara -works_at-> Acme`` to answer "where does my
sister work?", which cosine similarity structurally cannot.

Like the vector index in :mod:`.index`, the graph is a **derived, rebuildable**
cache: the note files are the source of truth, every write mirrors here, and
:func:`reindex` rebuilds the whole thing from disk. Traversal is a recursive CTE
(no graph engine, no new service), so the exact same shape ports to the Postgres
backend behind the ``postgres_backend`` seam.

``graph_nodes`` holds one row per resolved entity (keyed by a normalized slug);
``graph_edges`` holds directed ``subj -rel-> obj`` triples, each stamped with the
provenance ``note_name`` and optional ``valid_from``/``valid_to`` dates so a
fact that stopped being true can be excluded from time-aware recall.
"""

from __future__ import annotations

from datetime import date

from ..config import Settings, postgres_backend
from ..sqlite_util import open_db
from .locks import locked
from .store import Note, slugify

NODES_TABLE = "graph_nodes"
EDGES_TABLE = "graph_edges"

# Free text the user uses for themselves; all fold onto one "user" node so a
# triple about "the user" / "Omar" / "me" lands on the same entity.
_SELF_KEYS = frozenset({"user", "the-user", "me", "i", "myself"})
_SELF_KEY = "user"


def node_key(label: str) -> str:
    """Normalized key for an entity label (the graph's identity for a thing)."""
    key = slugify(label)
    return _SELF_KEY if key in _SELF_KEYS else key


def _connect(settings: Settings):
    conn = open_db(settings.graph_db_path, row_factory=False)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {NODES_TABLE} ("
        " key TEXT PRIMARY KEY, type TEXT DEFAULT '', label TEXT DEFAULT '')"
    )
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {EDGES_TABLE} ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " subj TEXT, rel TEXT, obj TEXT, note_name TEXT,"
        " valid_from TEXT DEFAULT '', valid_to TEXT DEFAULT '')"
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_edges_subj ON {EDGES_TABLE}(subj)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_edges_obj ON {EDGES_TABLE}(obj)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_edges_note ON {EDGES_TABLE}(note_name)")
    return conn


def _upsert_node(conn, label: str) -> str:
    key = node_key(label)
    conn.execute(
        f"INSERT INTO {NODES_TABLE}(key, label) VALUES (?, ?) "
        f"ON CONFLICT(key) DO UPDATE SET label = excluded.label",
        (key, label),
    )
    return key


@locked
def sync_note(settings: Settings, note: Note) -> None:
    """Mirror a note's triples into the graph (replacing its previous edges).

    Idempotent per note: all edges tagged with this note's name are dropped and
    re-inserted from ``note.relations``, so a re-saved/updated note never leaves
    stale edges behind. Nodes referenced by the new edges are upserted.
    """
    if pg := postgres_backend(settings):
        pg.sync_graph_note(settings, note)
        return
    conn = _connect(settings)
    try:
        conn.execute(f"DELETE FROM {EDGES_TABLE} WHERE note_name = ?", (note.name,))
        for rel in note.relations:
            subj = _upsert_node(conn, rel["subj"])
            obj = _upsert_node(conn, rel["obj"])
            conn.execute(
                f"INSERT INTO {EDGES_TABLE}"
                f"(subj, rel, obj, note_name, valid_from, valid_to)"
                f" VALUES (?, ?, ?, ?, ?, ?)",
                (subj, slugify(rel["rel"]), obj, note.name,
                 rel.get("valid_from", ""), rel.get("valid_to", "")),
            )
        conn.commit()
    finally:
        conn.close()


@locked
def remove(settings: Settings, name: str) -> None:
    """Drop every edge whose provenance is the note ``name`` (no-op if none)."""
    if pg := postgres_backend(settings):
        pg.remove_graph_note(settings, name)
        return
    conn = _connect(settings)
    try:
        conn.execute(f"DELETE FROM {EDGES_TABLE} WHERE note_name = ?", (name,))
        conn.commit()
    finally:
        conn.close()


def _valid_at(at_date: str) -> tuple[str, tuple]:
    """SQL fragment (and params) restricting to edges valid on ``at_date``.

    An edge is excluded only when a stamp definitively rules it out: a non-empty
    ``valid_to`` earlier than the date, or a non-empty ``valid_from`` later. A
    blank stamp is treated as open-ended (always valid), so undated facts — the
    common case — are never filtered out.
    """
    if not at_date:
        return "", ()
    frag = (
        " AND (valid_to = '' OR valid_to >= ?)"
        " AND (valid_from = '' OR valid_from <= ?)"
    )
    return frag, (at_date, at_date)


@locked
def neighbors(
    settings: Settings, keys: list[str], hops: int, at_date: str | None = None
) -> set[str]:
    """Node keys reachable from ``keys`` within ``hops`` (undirected, inclusive).

    Traversal ignores edge direction — a relationship is just as informative read
    backwards — so the connected neighborhood is gathered regardless of which way
    each triple was authored. ``at_date`` (default: today) drops edges whose
    validity window excludes it; pass ``""`` to ignore validity entirely.
    """
    if pg := postgres_backend(settings):
        return pg.graph_neighbors(settings, keys, hops, at_date)
    seeds = [node_key(k) for k in keys if k.strip()]
    if not seeds or hops < 0:
        return set()
    if at_date is None:
        at_date = date.today().isoformat()
    frag, extra = _valid_at(at_date)
    placeholders = ",".join("?" for _ in seeds)
    sql = (
        f"WITH RECURSIVE reach(node, depth) AS ("
        f"  SELECT key, 0 FROM {NODES_TABLE} WHERE key IN ({placeholders})"
        f"  UNION"
        f"  SELECT CASE WHEN e.subj = r.node THEN e.obj ELSE e.subj END, r.depth + 1"
        f"  FROM reach r JOIN {EDGES_TABLE} e"
        f"    ON (e.subj = r.node OR e.obj = r.node)"
        f"  WHERE r.depth < ?{frag}"
        f") SELECT DISTINCT node FROM reach"
    )
    conn = _connect(settings)
    try:
        rows = conn.execute(sql, (*seeds, hops, *extra)).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


@locked
def note_names_for(settings: Settings, node_keys: set[str]) -> list[str]:
    """Provenance note names of every edge touching any of ``node_keys``.

    These are the notes recall pulls in as extra candidates: the source facts
    that gave the neighborhood its edges.
    """
    if pg := postgres_backend(settings):
        return pg.graph_note_names(settings, node_keys)
    if not node_keys:
        return []
    placeholders = ",".join("?" for _ in node_keys)
    keys = list(node_keys)
    conn = _connect(settings)
    try:
        rows = conn.execute(
            f"SELECT DISTINCT note_name FROM {EDGES_TABLE} "
            f"WHERE subj IN ({placeholders}) OR obj IN ({placeholders})",
            (*keys, *keys),
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows if row[0]]


@locked
def resolve(settings: Settings, text: str) -> list[str]:
    """Node keys whose entity label is mentioned in ``text`` (recall's seeds).

    A lightweight surface-form match: a node matches when its label appears as a
    whole word (case-insensitive) in the query, or the query mentions the user
    ("my"/"me"/"i") — enough to seed traversal from the entities a question names
    without a second LLM call.
    """
    if pg := postgres_backend(settings):
        return pg.graph_resolve(settings, text)
    lowered = f" {text.lower()} "
    hits: list[str] = []
    conn = _connect(settings)
    try:
        rows = conn.execute(f"SELECT key, label FROM {NODES_TABLE}").fetchall()
    finally:
        conn.close()
    for key, label in rows:
        if key == _SELF_KEY:
            if any(f" {w} " in lowered for w in ("my", "me", "i", "mine", "myself")):
                hits.append(key)
            continue
        needle = (label or key).lower().strip()
        if needle and f" {needle} " in lowered:
            hits.append(key)
    return hits


@locked
def list_edges(settings: Settings) -> list[tuple[str, str, str, str]]:
    """All edges as ``(subj, rel, obj, note_name)`` — for tests/consolidation."""
    if pg := postgres_backend(settings):
        return pg.graph_list_edges(settings)
    conn = _connect(settings)
    try:
        return conn.execute(
            f"SELECT subj, rel, obj, note_name FROM {EDGES_TABLE} ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


@locked
def prune_orphans(settings: Settings, live_names: set[str]) -> int:
    """Delete edges whose provenance note no longer exists. Returns count dropped.

    Called from consolidation so a forgotten/renamed note can't leave dangling
    edges that would keep surfacing its facts.
    """
    if pg := postgres_backend(settings):
        return pg.prune_graph_orphans(settings, live_names)
    conn = _connect(settings)
    try:
        names = {
            row[0] for row in conn.execute(f"SELECT note_name FROM {EDGES_TABLE}")
        }
        stale = [n for n in names if n not in live_names]
        for name in stale:
            conn.execute(f"DELETE FROM {EDGES_TABLE} WHERE note_name = ?", (name,))
        # Drop nodes that no edge references any more.
        conn.execute(
            f"DELETE FROM {NODES_TABLE} WHERE key NOT IN "
            f"(SELECT subj FROM {EDGES_TABLE} UNION SELECT obj FROM {EDGES_TABLE})"
        )
        conn.commit()
        return len(stale)
    finally:
        conn.close()


@locked
def reindex(settings: Settings) -> int:
    """Rebuild the whole graph from the markdown notes. Returns edges written.

    The self-healing path (mirrors :func:`.index.reindex`): the graph tables are
    cleared and repopulated purely from each note's ``relations`` frontmatter, so
    a corrupt or deleted ``graph.db`` — or a hand-edited note — reconciles on the
    next pass. No LLM re-extraction: the triples already live on disk.
    """
    if pg := postgres_backend(settings):
        return pg.reindex_graph(settings)
    from . import store

    notes = store.list_notes(settings)
    conn = _connect(settings)
    try:
        conn.execute(f"DELETE FROM {EDGES_TABLE}")
        conn.execute(f"DELETE FROM {NODES_TABLE}")
        written = 0
        for note in notes:
            for rel in note.relations:
                subj = _upsert_node(conn, rel["subj"])
                obj = _upsert_node(conn, rel["obj"])
                conn.execute(
                    f"INSERT INTO {EDGES_TABLE}"
                    f"(subj, rel, obj, note_name, valid_from, valid_to)"
                    f" VALUES (?, ?, ?, ?, ?, ?)",
                    (subj, slugify(rel["rel"]), obj, note.name,
                     rel.get("valid_from", ""), rel.get("valid_to", "")),
                )
                written += 1
        conn.commit()
        return written
    finally:
        conn.close()
