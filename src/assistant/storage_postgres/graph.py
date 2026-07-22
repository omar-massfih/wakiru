"""Knowledge-graph nodes + edges for the Postgres backend.

The Postgres twin of :mod:`assistant.memory.graph`: same two-table model
(``assistant_graph_nodes`` / ``assistant_graph_edges``) and the same undirected
recursive-CTE traversal, so multi-hop recall behaves identically whichever
backend is configured. No graph extension (Apache AGE) is required — plain
tables plus a ``WITH RECURSIVE`` walk keep this portable to managed Neon.
"""

from __future__ import annotations

from datetime import date

from ..config import Settings
from ..memory.graph import node_key
from ..memory.store import Note, slugify
from .core import _schema_done, _schema_mark, connect


def ensure_graph_schema(settings: Settings) -> None:
    if _schema_done(settings, "graph"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_graph_nodes (
              key TEXT PRIMARY KEY,
              type TEXT NOT NULL DEFAULT '',
              label TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_graph_edges (
              id BIGSERIAL PRIMARY KEY,
              subj TEXT NOT NULL,
              rel TEXT NOT NULL,
              obj TEXT NOT NULL,
              note_name TEXT NOT NULL,
              valid_from TEXT NOT NULL DEFAULT '',
              valid_to TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_subj "
            "ON assistant_graph_edges(subj)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_obj "
            "ON assistant_graph_edges(obj)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_note "
            "ON assistant_graph_edges(note_name)"
        )
    _schema_mark(settings, "graph")


def _upsert_node(conn, label: str) -> str:
    key = node_key(label)
    conn.execute(
        "INSERT INTO assistant_graph_nodes(key, label) VALUES (%s, %s) "
        "ON CONFLICT(key) DO UPDATE SET label = excluded.label",
        (key, label),
    )
    return key


def sync_graph_note(settings: Settings, note: Note) -> None:
    ensure_graph_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "DELETE FROM assistant_graph_edges WHERE note_name = %s", (note.name,)
        )
        for rel in note.relations:
            subj = _upsert_node(conn, rel["subj"])
            obj = _upsert_node(conn, rel["obj"])
            conn.execute(
                "INSERT INTO assistant_graph_edges"
                "(subj, rel, obj, note_name, valid_from, valid_to)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (subj, slugify(rel["rel"]), obj, note.name,
                 rel.get("valid_from", ""), rel.get("valid_to", "")),
            )


def remove_graph_note(settings: Settings, name: str) -> None:
    ensure_graph_schema(settings)
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_graph_edges WHERE note_name = %s", (name,))


def graph_neighbors(
    settings: Settings, keys: list[str], hops: int, at_date: str | None = None
) -> set[str]:
    seeds = [node_key(k) for k in keys if k.strip()]
    if not seeds or hops < 0:
        return set()
    if at_date is None:
        at_date = date.today().isoformat()
    valid = ""
    params: list = [seeds, hops]
    if at_date:
        valid = (
            " AND (e.valid_to = '' OR e.valid_to >= %s)"
            " AND (e.valid_from = '' OR e.valid_from <= %s)"
        )
        params += [at_date, at_date]
    sql = (
        "WITH RECURSIVE reach(node, depth) AS ("
        "  SELECT key, 0 FROM assistant_graph_nodes WHERE key = ANY(%s)"
        "  UNION"
        "  SELECT CASE WHEN e.subj = r.node THEN e.obj ELSE e.subj END, r.depth + 1"
        "  FROM reach r JOIN assistant_graph_edges e"
        "    ON (e.subj = r.node OR e.obj = r.node)"
        "  WHERE r.depth < %s" + valid +
        ") SELECT DISTINCT node FROM reach"
    )
    ensure_graph_schema(settings)
    with connect(settings) as conn:
        rows = conn.execute(sql, params).fetchall()
    return {row[0] for row in rows}


def graph_note_names(settings: Settings, node_keys: set[str]) -> list[str]:
    if not node_keys:
        return []
    ensure_graph_schema(settings)
    keys = list(node_keys)
    with connect(settings) as conn:
        rows = conn.execute(
            "SELECT DISTINCT note_name FROM assistant_graph_edges "
            "WHERE subj = ANY(%s) OR obj = ANY(%s)",
            (keys, keys),
        ).fetchall()
    return [row[0] for row in rows if row[0]]


def graph_resolve(settings: Settings, text: str) -> list[str]:
    lowered = f" {text.lower()} "
    ensure_graph_schema(settings)
    with connect(settings) as conn:
        rows = conn.execute("SELECT key, label FROM assistant_graph_nodes").fetchall()
    hits: list[str] = []
    for key, label in rows:
        if key == "user":
            if any(f" {w} " in lowered for w in ("my", "me", "i", "mine", "myself")):
                hits.append(key)
            continue
        needle = (label or key).lower().strip()
        if needle and f" {needle} " in lowered:
            hits.append(key)
    return hits


def graph_list_edges(settings: Settings) -> list[tuple[str, str, str, str]]:
    ensure_graph_schema(settings)
    with connect(settings) as conn:
        return [
            (r[0], r[1], r[2], r[3])
            for r in conn.execute(
                "SELECT subj, rel, obj, note_name FROM assistant_graph_edges ORDER BY id"
            ).fetchall()
        ]


def prune_graph_orphans(settings: Settings, live_names: set[str]) -> int:
    ensure_graph_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "DELETE FROM assistant_graph_edges WHERE NOT (note_name = ANY(%s)) "
            "RETURNING note_name",
            (list(live_names),),
        )
        # Count distinct orphaned note names, not edge rows, so the returned
        # tally matches the SQLite twin (memory.graph.prune_orphans).
        dropped = len({r[0] for r in cur.fetchall()})
        conn.execute(
            "DELETE FROM assistant_graph_nodes WHERE key NOT IN "
            "(SELECT subj FROM assistant_graph_edges "
            " UNION SELECT obj FROM assistant_graph_edges)"
        )
    return dropped


def reindex_graph(settings: Settings) -> int:
    from ..memory import store

    ensure_graph_schema(settings)
    notes = store.list_notes(settings)
    with connect(settings) as conn:
        conn.execute("DELETE FROM assistant_graph_edges")
        conn.execute("DELETE FROM assistant_graph_nodes")
        written = 0
        for note in notes:
            for rel in note.relations:
                subj = _upsert_node(conn, rel["subj"])
                obj = _upsert_node(conn, rel["obj"])
                conn.execute(
                    "INSERT INTO assistant_graph_edges"
                    "(subj, rel, obj, note_name, valid_from, valid_to)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    (subj, slugify(rel["rel"]), obj, note.name,
                     rel.get("valid_from", ""), rel.get("valid_to", "")),
                )
                written += 1
    return written
