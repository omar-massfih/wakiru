"""Knowledge-graph memory tests — triples, traversal, recall augmentation, rebuild.

Like ``test_memory.py`` these run offline: embeddings are faked with the same
deterministic bag-of-words vector, so only the sqlite-vec index, the file store,
and the real graph tables run. The graph is what lets recall answer a multi-hop
question ("where does my sister work?") whose answer no single note phrases like
the query.
"""

from __future__ import annotations

import math
import re
import zlib

import pytest

from assistant.config import Settings
from assistant.memory import graph, index, learn, recall, store
from assistant.memory.store import Note


def _fake_embed(texts: list[str], prefix: str = "", settings=None) -> list[list[float]]:
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_auto_memory=False,
        dedup_threshold=0.8,
    )


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    monkeypatch.setattr("assistant.memory.embeddings._embed", _fake_embed)


# --- store round-trip of relations --------------------------------------- #


def test_relations_roundtrip_in_frontmatter(settings) -> None:
    note = Note(
        name="sister-job",
        description="Sara works at Acme",
        body="The user's sister Sara works at Acme.",
        relations=[
            {"subj": "user", "rel": "sister", "obj": "Sara"},
            {"subj": "Sara", "rel": "works_at", "obj": "Acme"},
        ],
    )
    path = store.write_note(settings, note)
    assert "relations:" in path.read_text(encoding="utf-8")
    back = store.read_note(path)
    assert back.relations == note.relations


def test_relations_normalized_drops_malformed(settings) -> None:
    note = Note(
        name="n", description="d", body="b",
        relations=[
            {"subj": "user", "rel": "sister", "obj": "Sara"},
            {"subj": "user", "rel": "", "obj": "x"},   # blank rel -> dropped
            "not-a-dict",                               # wrong type -> dropped
        ],
    )
    assert note.relations == [{"subj": "user", "rel": "sister", "obj": "Sara"}]


def test_empty_relations_not_serialized(settings) -> None:
    path = store.write_note(settings, Note(name="n", description="d", body="b"))
    assert "relations:" not in path.read_text(encoding="utf-8")


# --- graph sync + traversal ---------------------------------------------- #


def _seed_sister_graph(settings) -> None:
    learn.save_memory(
        settings,
        body="The user's sister is Sara.",
        description="user's sister Sara",
        relations=[{"subj": "user", "rel": "sister", "obj": "Sara"}],
    )
    learn.save_memory(
        settings,
        body="Sara works at Acme.",
        description="Sara works at Acme",
        relations=[{"subj": "Sara", "rel": "works_at", "obj": "Acme"}],
    )


def test_save_mirrors_edges(settings) -> None:
    _seed_sister_graph(settings)
    edges = {(s, r, o) for s, r, o, _n in graph.list_edges(settings)}
    assert ("user", "sister", "sara") in edges
    assert ("sara", "works-at", "acme") in edges


def test_neighbors_multi_hop(settings) -> None:
    _seed_sister_graph(settings)
    one_hop = graph.neighbors(settings, ["user"], hops=1)
    assert "sara" in one_hop and "acme" not in one_hop
    two_hop = graph.neighbors(settings, ["user"], hops=2)
    assert {"sara", "acme"} <= two_hop


def test_neighbors_honors_validity(settings) -> None:
    learn.save_memory(
        settings,
        body="The user lived in Oslo until 2026.",
        description="user lived in Oslo",
        relations=[{"subj": "user", "rel": "lives_in", "obj": "Oslo",
                    "valid_to": "2026-01-01"}],
    )
    # As of today (well past valid_to), the expired edge is excluded.
    assert graph.neighbors(settings, ["user"], hops=1, at_date="2027-01-01") == {"user"}
    # Ignoring validity, the edge is present.
    assert "oslo" in graph.neighbors(settings, ["user"], hops=1, at_date="")


def test_resolve_finds_named_entities(settings) -> None:
    _seed_sister_graph(settings)
    assert "sara" in graph.resolve(settings, "Tell me about Sara please")
    assert "user" in graph.resolve(settings, "where does my sister work")


# --- recall augmentation (the headline capability) ----------------------- #


def test_multi_hop_recall(settings) -> None:
    _seed_sister_graph(settings)
    # The query shares no salient words with "Sara works at Acme"; a pure vector
    # lookup would miss it. Graph traversal user->sister->Sara->works_at->Acme
    # pulls the provenance note in.
    results = recall.search_memory(settings, "where does my sister work")
    names = {note.name for note, _ in results}
    assert any("acme" in n for n in names), names


def test_graph_disabled_is_pure_vector(settings) -> None:
    off = settings.model_copy(update={"enable_graph_memory": False})
    _seed_sister_graph(off)
    results = recall.search_memory(off, "where does my sister work")
    names = {note.name for note, _ in results}
    assert not any("acme" in n for n in names), names


# --- forget + rebuild ----------------------------------------------------- #


def test_forget_drops_edges(settings) -> None:
    _seed_sister_graph(settings)
    learn.forget_memory(settings, "sara-works-at-acme")
    edges = {(s, r, o) for s, r, o, _n in graph.list_edges(settings)}
    assert ("sara", "works-at", "acme") not in edges
    assert ("user", "sister", "sara") in edges


def test_reindex_rebuilds_from_files(settings) -> None:
    _seed_sister_graph(settings)
    before = sorted(graph.list_edges(settings))
    # Nuke the derived graph DB; the markdown notes are the source of truth.
    settings.graph_db_path.unlink()
    assert graph.list_edges(settings) == []
    graph.reindex(settings)
    assert sorted(graph.list_edges(settings)) == before


def test_reindex_picks_up_hand_edit(settings) -> None:
    note = store.write_note(
        settings,
        Note(name="brother", description="user's brother Ola", body="Brother is Ola.",
             relations=[{"subj": "user", "rel": "brother", "obj": "Ola"}]),
    )
    del note  # written to disk; graph not yet synced for this direct store write
    graph.reindex(settings)
    edges = {(s, r, o) for s, r, o, _n in graph.list_edges(settings)}
    assert ("user", "brother", "ola") in edges


def test_consolidation_prunes_orphan_edges(settings) -> None:
    from assistant.memory import consolidate

    _seed_sister_graph(settings)
    # Delete the note out from under the graph without going through forget.
    store.delete_note(settings, "sara-works-at-acme")
    index.remove(settings, "sara-works-at-acme")
    consolidate.consolidate_memory(settings, include_llm=False)
    edges = {(s, r, o) for s, r, o, _n in graph.list_edges(settings)}
    assert ("sara", "works-at", "acme") not in edges
