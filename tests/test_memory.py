"""Memory tests — store round-trip, kinds, vector index, dedup, recall, reconcile.

Embeddings are faked with a deterministic bag-of-words vector so these stay fast
and offline; only the sqlite-vec index and the file store run for real. The fake
is patched at ``embeddings._embed`` — the single seam every embed wrapper
(passage/query/one) routes through — so it ignores the e5 query:/passage: prefix.
"""

from __future__ import annotations

import math
import re
import zlib

import pytest

from assistant.config import Settings
from assistant.memory import consolidate, embeddings, index, learn, recall, store
from assistant.memory.store import Note


def _fake_embed(texts: list[str], prefix: str = "", settings=None) -> list[list[float]]:
    """Normalized term-frequency vectors: overlap in words -> high cosine."""
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            # crc32 is stable across processes; builtin hash() is randomized.
            v[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Thresholds tuned to the fake bag-of-words embedder (wider, lower spread
    # than the real e5 model whose production defaults are higher).
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_auto_memory=False,
        dedup_threshold=0.8,
        forget_threshold=0.4,
        forget_ambiguity_margin=0.1,
    )


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    monkeypatch.setattr("assistant.memory.embeddings._embed", _fake_embed)


# --- store ---------------------------------------------------------------- #


def test_note_roundtrip(settings) -> None:
    note = Note(
        name="user-name",
        description="The user is Omar",
        body="The user's name is Omar.",
        kind="semantic",
        salience=0.9,
        tags=["identity"],
        source="t-1",
    )
    path = store.write_note(settings, note)
    assert path.parent.name == "semantic"
    back = store.read_note(path)
    assert back.name == "user-name"
    assert back.kind == "semantic"
    assert back.salience == 0.9
    assert back.tags == ["identity"]
    assert back.source == "t-1"


def test_kind_directories_and_legacy(settings) -> None:
    assert store.write_note(
        settings, Note(name="a", description="d", body="b", kind="procedural")
    ).parent.name == "procedural"
    assert store.write_note(
        settings, Note(name="e", description="d", body="b", kind="episodic")
    ).parent.name == "episodic"
    # Legacy "fact"/"learning" map onto semantic/procedural.
    assert Note(name="x", description="d", body="b", kind="fact").kind == "semantic"
    assert Note(name="y", description="d", body="b", kind="learning").kind == "procedural"


def test_regenerate_index_grouped_by_kind(settings) -> None:
    store.write_note(settings, Note(name="fact1", description="a fact", body="A.", kind="semantic"))
    store.write_note(settings, Note(name="how1", description="a method", body="B.", kind="procedural"))
    store.regenerate_index(settings)
    text = store.read_index(settings)
    assert "## Semantic" in text and "## Procedural" in text
    assert "a fact" in text and "a method" in text


def test_unique_name_avoids_clobber(settings) -> None:
    store.write_note(settings, Note(name="dup", description="d", body="one"))
    assert store.unique_name(settings, "dup") == "dup-2"
    assert store.unique_name(settings, "dup", keep="dup") == "dup"  # same note is fine
    assert store.unique_name(settings, "fresh") == "fresh"


# --- index + recall ------------------------------------------------------- #


def test_save_and_recall(settings) -> None:
    learn.save_memory(settings, body="The user prefers Norwegian language replies.")
    results = recall.search_memory(settings, "Norwegian language replies preference")
    assert results, "expected the Norwegian preference to be recalled"
    assert "norwegian" in results[0][0].body.lower()


def test_recall_below_threshold_returns_nothing(settings) -> None:
    learn.save_memory(settings, body="The user prefers replies in Norwegian.")
    assert recall.search_memory(settings, "quarterly budget spreadsheet totals") == []


def test_dedup_updates_in_place(settings) -> None:
    learn.save_memory(settings, body="The user prefers replies in Norwegian.")
    learn.save_memory(settings, body="The user prefers replies in Norwegian please.")
    assert len(store.list_notes(settings)) == 1


def test_distinct_notes_do_not_clobber(settings) -> None:
    # Two unrelated facts whose descriptions slugify identically must coexist.
    learn.save_memory(settings, body="x", description="favorite thing")
    learn.save_memory(settings, body="y totally different content here", description="favorite thing")
    names = {n.name for n in store.list_notes(settings)}
    assert names == {"favorite-thing", "favorite-thing-2"}


def test_episode_does_not_swallow_semantic_save(settings) -> None:
    # An episodic trace and a semantic fact can be near-identical in embedding
    # space; the semantic save must form its own note, not dedup into the episode.
    learn.record_episode(settings, "My name is Omar and I live in Oslo", "Noted.")
    learn.save_memory(settings, body="The user lives in Oslo.", kind="semantic")
    kinds = sorted(n.kind for n in store.list_notes(settings))
    assert kinds == ["episodic", "semantic"]


def test_kind_change_leaves_no_stale_file(settings) -> None:
    note = learn.save_memory(settings, body="Deploy with uv.", kind="semantic")
    learn.revise_memory(settings, note.name, kind="procedural")
    paths = [str(store.note_path(settings, n).parent.name) for n in store.list_notes(settings)]
    assert paths == ["procedural"]  # the old semantic/ copy is gone
    assert len(store.list_notes(settings)) == 1


def test_reuse_reinforcement_reranks(settings) -> None:
    # "alpha" and "beta" are equidistant from the query "alpha beta"; the one
    # recalled more often should rank first.
    learn.save_memory(settings, body="alpha")
    learn.save_memory(settings, body="beta")
    index.bump_recall(settings, ["alpha", "alpha", "alpha"])
    results = recall.search_memory(settings, "alpha beta")
    assert results[0][0].name == "alpha"


def test_recall_context_reinforces(settings) -> None:
    learn.save_memory(settings, body="The user lives in Oslo.")
    recall.recall_context(settings, "Where does the user live?")
    stats = index.get_stats(settings, store.list_notes(settings)[0].name)
    assert stats is not None and stats[0] >= 1  # recall_count bumped


# --- reconciliation (update / forget) ------------------------------------- #


def test_revise_memory_supersedes_in_place(settings) -> None:
    note = learn.save_memory(settings, body="The user lives in Oslo.", description="Where the user lives")
    revised = learn.revise_memory(settings, note.name, body="The user lives in Bergen.")
    assert revised is not None
    notes = store.list_notes(settings)
    assert len(notes) == 1
    assert "bergen" in notes[0].body.lower()
    assert notes[0].created == note.created  # identity preserved


def test_revise_preserves_reinforcement_counters(settings) -> None:
    # The index is the authoritative counter store; an in-place update must not
    # reset recall_count/last_recalled to the (lagging) file values.
    note = learn.save_memory(settings, body="The user lives in Oslo.", description="home")
    index.bump_recall(settings, [note.name, note.name])
    learn.revise_memory(settings, note.name, body="The user lives in Bergen.")
    stats = index.get_stats(settings, note.name)
    assert stats is not None and stats[0] == 2  # counters survived the update


def test_forget_by_name_and_by_query(settings) -> None:
    learn.save_memory(settings, body="The user's favorite color is teal.", description="favorite color")
    assert learn.forget_memory(settings, "favorite-color") is not None  # by name
    learn.save_memory(settings, body="The user's favorite color is teal.")
    assert learn.forget_memory(settings, "favorite color teal") is not None  # by query
    assert store.list_notes(settings) == []


def test_forget_no_match_returns_none(settings) -> None:
    learn.save_memory(settings, body="The user's favorite color is teal.")
    assert learn.forget_memory(settings, "quarterly budget spreadsheet") is None
    assert store.list_notes(settings)


def test_dedup_absorb_preserves_tags(settings) -> None:
    # Hand-added tags must survive a restatement absorbing the note in place.
    note = learn.save_memory(settings, body="The user prefers replies in Norwegian.")
    tagged = store.find_note(settings, note.name)
    assert tagged is not None
    tagged.tags = ["language"]
    store.write_note(settings, tagged)
    learn.save_memory(settings, body="The user prefers replies in Norwegian please.")
    notes = store.list_notes(settings)
    assert len(notes) == 1 and notes[0].tags == ["language"]


def test_forget_by_exact_name_refuses_episodic(settings) -> None:
    # Episodes are a log pruned by consolidation; even an exact-name forget
    # (e.g. an LLM-hallucinated name) must not delete one.
    episode = learn.record_episode(settings, "went hiking in the mountains today", "Nice!")
    assert episode is not None
    assert learn.forget_memory(settings, episode.name) is None
    assert len(store.list_notes(settings)) == 1


def test_format_memories_sees_durables_past_episodes(settings) -> None:
    # The current turn's own episodic trace near-duplicates the query and tops a
    # plain top-k; the reconciliation list must still surface durable notes.
    settings.recall_top_k = 1
    settings.recall_candidate_multiplier = 3
    learn.save_memory(settings, body="The user lives in Oslo right now.", kind="semantic")
    learn.record_episode(settings, "where does the user live right now", "In Oslo.")
    text = learn._format_memories(settings, "where does the user live right now")
    assert "The user lives in Oslo" in text


# --- reindex (files are the source of truth) ------------------------------ #


def test_reindex_picks_up_new_file_and_drops_deleted(settings) -> None:
    learn.save_memory(settings, body="The user likes hiking.")
    # Hand-write a note straight to disk (no index entry yet).
    store.write_note(settings, Note(name="manual", description="added by hand", body="A hand-added fact."))
    index.reindex(settings)
    assert recall.search_memory(settings, "hand-added fact")

    store.delete_note(settings, "manual")
    index.reindex(settings)
    assert not recall.search_memory(settings, "hand-added fact")


def test_reindex_skips_unchanged_notes(settings, monkeypatch) -> None:
    learn.save_memory(settings, body="The user likes hiking.")
    learn.save_memory(settings, body="The user plays chess.")

    calls: list[list[str]] = []

    def counting(texts, prefix="", settings=None):
        calls.append(list(texts))
        return _fake_embed(texts, prefix, settings)

    monkeypatch.setattr("assistant.memory.embeddings._embed", counting)
    assert index.reindex(settings) == 2
    assert calls == []  # a routine restart embeds nothing

    # A hand-edited file changes its hash and must be re-embedded — only it.
    note = store.find_note(settings, "the-user-likes-hiking")
    assert note is not None
    note.body = "The user likes hiking and skiing."
    store.write_note(settings, note)
    assert index.reindex(settings) == 2
    assert len(calls) == 1 and len(calls[0]) == 1
    assert recall.search_memory(settings, "hiking and skiing")


def test_reindex_on_model_change_preserves_counters(settings, tmp_path) -> None:
    learn.save_memory(settings, body="The user plays chess.")
    name = store.list_notes(settings)[0].name
    index.bump_recall(settings, [name, name])
    # A different model name forces a rebuild; counters must survive.
    changed = Settings(
        memory_dir=settings.memory_dir,
        enable_auto_memory=False,
        embedding_model="some/other-model",
    )
    index.reindex(changed)
    stats = index.get_stats(changed, name)
    assert stats is not None and stats[0] == 2


# --- LLM-driven update (save + update + forget in one pass) ---------------- #


def test_update_memory_applies_save_update_forget(tmp_path, monkeypatch) -> None:
    # episodic_min_chars=0: the short "user text" message must still leave a trace.
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_auto_memory=True,
        episodic_min_chars=0,
    )
    learn.save_memory(settings, body="The user's favorite color is teal.", description="favorite color")
    home = learn.save_memory(settings, body="The user lives in Oslo.", description="Where the user lives")

    canned = (
        '[{"op": "save", "kind": "semantic", "description": "Lives in Oslo",'
        ' "body": "The user lives in Oslo."},'
        f' {{"op": "update", "name": "{home.name}", "description": "Where the user lives",'
        ' "body": "The user now lives in Bergen."},'
        ' {"op": "forget", "name": "favorite-color"}]'
    )
    monkeypatch.setattr("assistant.memory.learn.run_codex", lambda *a, **k: canned)

    applied = learn.update_memory(settings, "user text", "assistant text", source="t-9")

    bodies = " ".join(n.body.lower() for n in store.list_notes(settings))
    assert "bergen" in bodies and "teal" not in bodies
    assert any(s.startswith("updated:") for s in applied)
    assert any(s.startswith("forgot:") for s in applied)
    # An episodic trace was recorded for the turn.
    assert any(n.kind == "episodic" for n in store.list_notes(settings))


def test_update_memory_disabled_is_noop(tmp_path, monkeypatch) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), enable_auto_memory=False)
    monkeypatch.setattr(
        "assistant.memory.learn.run_codex",
        lambda *a, **k: pytest.fail("run_codex must not be called when disabled"),
    )
    assert learn.update_memory(settings, "hi", "hello") == []


# --- bounded index view ----------------------------------------------------- #


def test_index_view_caps_per_kind(settings) -> None:
    settings.context_index_max_per_kind = {"semantic": 2, "procedural": -1, "episodic": 0}
    for body, sal in [("alpha", 0.9), ("bravo", 0.8), ("charlie", 0.2), ("delta", 0.1)]:
        learn.save_memory(settings, body=body, salience=sal)
    view = recall.build_index_view(settings)
    assert "Memory: 4 semantic." in view  # stats line reports the true total
    assert "**alpha**" in view and "**bravo**" in view
    assert "charlie" not in view and "delta" not in view
    assert "Showing the most valuable" in view


def test_index_view_omits_episodic_but_recall_still_finds_them(settings) -> None:
    learn.record_episode(settings, "went hiking with Torvald in the hills", "Nice!")
    view = recall.build_index_view(settings)
    assert "1 episodic" in view  # counted in the stats line…
    assert "hiking" not in view.lower()  # …but never listed
    assert recall.search_memory(settings, "hiking Torvald hills")


def test_index_view_unlimited_lists_everything(settings) -> None:
    settings.context_index_max_per_kind = {"semantic": -1, "procedural": -1, "episodic": -1}
    for body in ["alpha", "bravo", "charlie"]:
        learn.save_memory(settings, body=body)
    view = recall.build_index_view(settings)
    assert all(f"**{name}**" in view for name in ["alpha", "bravo", "charlie"])
    assert "Showing the most valuable" not in view


# --- episodic gating -------------------------------------------------------- #


def test_short_message_records_no_episode(settings) -> None:
    assert learn.record_episode(settings, "hi", "Hello!") is None
    assert store.list_notes(settings) == []


def test_duplicate_exchange_records_one_episode(settings) -> None:
    msg = "I always deploy my projects with uv"
    assert learn.record_episode(settings, msg, "Noted.") is not None
    assert learn.record_episode(settings, msg, "Noted.") is None
    assert len(store.list_notes(settings)) == 1


def test_distinct_exchanges_record_two_episodes(settings) -> None:
    learn.record_episode(settings, "went hiking in the mountains today", "Nice!")
    learn.record_episode(settings, "planning a chess tournament for friday", "Good luck!")
    assert len(store.list_notes(settings)) == 2


# --- cross-kind dedup + safer forget ---------------------------------------- #


def test_cross_kind_dedup_updates_durable_note(settings) -> None:
    learn.save_memory(settings, body="The user deploys projects with uv.", kind="semantic")
    learn.save_memory(settings, body="The user deploys projects with uv.", kind="procedural")
    notes = store.list_notes(settings)
    assert len(notes) == 1  # absorbed, and no stale semantic/ file left behind
    assert notes[0].kind == "procedural"


def test_cross_kind_distinct_notes_stay_separate(settings) -> None:
    learn.save_memory(settings, body="The user lives in Oslo.", kind="semantic")
    learn.save_memory(settings, body="Deploy the website with uv run.", kind="procedural")
    assert len(store.list_notes(settings)) == 2


def test_fuzzy_forget_ambiguous_is_noop(settings) -> None:
    # Both notes match "favorite color" equally well — deleting nothing beats
    # deleting the wrong one.
    learn.save_memory(settings, body="The user's favorite color is teal.", description="color teal")
    learn.save_memory(settings, body="The user's favorite color is red.", description="color red")
    assert learn.forget_memory(settings, "favorite color") is None
    assert len(store.list_notes(settings)) == 2


def test_fuzzy_forget_clear_winner_deletes(settings) -> None:
    learn.save_memory(settings, body="The user's favorite color is teal.", description="color teal")
    learn.save_memory(settings, body="The user plays chess on sundays.", description="chess habit")
    deleted = learn.forget_memory(settings, "favorite color teal")
    assert deleted is not None and "teal" in deleted.body
    assert {n.name for n in store.list_notes(settings)} == {"chess-habit"}


# --- persistent turn counter ------------------------------------------------ #


def test_turn_counter_persists_across_connections(settings) -> None:
    # Each call opens a fresh SQLite connection — surviving across calls is the
    # same property as surviving a server restart.
    assert index.bump_turn_counter(settings) == 1
    assert index.bump_turn_counter(settings) == 2
    assert index.bump_turn_counter(settings) == 3


# --- consolidation -------------------------------------------------------- #


def test_consolidate_prunes_and_promotes(tmp_path, monkeypatch) -> None:
    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_auto_memory=True,
        episodic_max_age_days=30,
    )
    # A stale episode (older than the horizon) should be pruned.
    old = Note(
        name="old-ep",
        description="ancient",
        body="Something old.",
        kind="episodic",
        created="2000-01-01",
        updated="2000-01-01",
    )
    store.write_note(settings, old)
    index.upsert(
        settings, old.name, str(store.note_path(settings, old)), old.description,
        embeddings.embed_one(old.index_text, settings),
        kind="episodic", updated="2000-01-01",
    )
    # A recent episode the LLM will "promote".
    learn.record_episode(settings, "I always deploy with uv", "Noted.")

    canned = (
        '[{"op": "save", "kind": "procedural", "description": "Deploys with uv",'
        ' "body": "The user deploys with uv."}]'
    )
    monkeypatch.setattr("assistant.memory.consolidate.run_codex", lambda *a, **k: canned)

    summary = consolidate.consolidate_memory(settings)
    assert summary["pruned_episodes"] >= 1
    names = {n.name for n in store.list_notes(settings)}
    assert "old-ep" not in names
    assert any(n.kind == "procedural" for n in store.list_notes(settings))


def test_consolidate_evicts_beyond_kind_cap(settings) -> None:
    settings.max_notes_per_kind = {"semantic": 3, "procedural": 0, "episodic": 0}
    for body, sal in [("alpha", 0.9), ("bravo", 0.8), ("charlie", 0.7), ("delta", 0.2), ("echo", 0.1)]:
        learn.save_memory(settings, body=body, salience=sal)
    summary = consolidate.consolidate_memory(settings)
    assert summary["evicted"] == 2
    assert {n.name for n in store.list_notes(settings)} == {"alpha", "bravo", "charlie"}
    assert not recall.search_memory(settings, "delta")  # gone from the index too


def test_eviction_keeps_recalled_note_over_equal_salience(settings) -> None:
    settings.max_notes_per_kind = {"semantic": 1, "procedural": 0, "episodic": 0}
    learn.save_memory(settings, body="alpha", salience=0.5)
    learn.save_memory(settings, body="bravo", salience=0.5)
    index.bump_recall(settings, ["bravo"])  # proving useful protects a memory
    consolidate.consolidate_memory(settings)
    assert {n.name for n in store.list_notes(settings)} == {"bravo"}


def test_consolidate_decays_unrecalled_durable(settings) -> None:
    stale = Note(
        name="stale-fact",
        description="an old fact",
        body="Old fact body.",
        kind="semantic",
        salience=0.5,
        created="2000-01-01",
        updated="2000-01-01",
    )
    store.write_note(settings, stale)
    index.upsert(
        settings, stale.name, str(store.note_path(settings, stale)), stale.description,
        embeddings.embed_one(stale.index_text, settings),
        kind="semantic", salience=0.5, updated="2000-01-01",
    )
    fresh = learn.save_memory(settings, body="a fresh recalled fact", salience=0.5)
    index.bump_recall(settings, [fresh.name])

    summary = consolidate.consolidate_memory(settings)
    assert summary["durable_decayed"] == 1

    effective = {row[0]: row[3] for row in index.list_entries(settings)}
    assert effective["stale-fact"] < 0.5  # decayed in the index (ranking/eviction)
    assert effective[fresh.name] == 0.5  # recall history exempts a note
    on_disk = store.find_note(settings, "stale-fact")
    assert on_disk is not None and on_disk.salience == 0.5  # file keeps the base
