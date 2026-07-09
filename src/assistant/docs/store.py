"""SQLite + sqlite-vec store for ingested documents.

A document is chunked, each chunk is embedded with the same local model the brain
uses (:mod:`assistant.memory.embeddings`), and the chunk vectors live in a
``vec0`` virtual table — the same machinery as :mod:`assistant.memory.index`, but
in its own ``docs.db`` so document chunks never mix with durable memory notes.
Recall goes through :func:`search_chunks` (cosine nearest-neighbour), so "what did
I write about X" is answered from the closest chunks rather than the whole corpus.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import sqlite3

import sqlite_vec

from ..config import Settings
from ..memory.embeddings import embed_passages, embed_query

_VEC_TABLE = "chunk_vec"


@dataclass
class Document:
    id: str
    title: str
    text: str
    added: str = ""
    chunks: int = 0


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    text: str
    similarity: float = 0.0


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.docs_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        " id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL,"
        " added TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL,"
        " ord INTEGER NOT NULL, text TEXT NOT NULL)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def _ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
    if row is not None:
        return
    conn.execute(
        f"CREATE VIRTUAL TABLE {_VEC_TABLE} USING"
        f" vec0(embedding float[{dim}] distance_metric=cosine)"
    )
    conn.execute("INSERT INTO meta(key, value) VALUES ('dim', ?)", (str(dim),))


def chunk_text(text: str, target_chars: int) -> list[str]:
    """Split ``text`` into chunks of about ``target_chars``, breaking on blank
    lines first so paragraphs stay intact, then packing them up to the target."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    # A single paragraph longer than the target is hard-split so no chunk is huge.
    out: list[str] = []
    for c in chunks:
        while len(c) > target_chars * 2:
            out.append(c[: target_chars * 2])
            c = c[target_chars * 2 :]
        out.append(c)
    return out or ([text.strip()] if text.strip() else [])


def add_document(settings: Settings, title: str, text: str) -> Document:
    """Ingest a document: store it, chunk it, embed each chunk, index the vectors."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        pieces = chunk_text(text, settings.docs_chunk_chars)
        vectors = embed_passages(pieces, settings) if pieces else []
        if len(vectors) != len(pieces):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(pieces)} chunks"
            )
        return storage_postgres.add_document(settings, title, text, pieces, vectors)
    doc = Document(
        id=uuid.uuid4().hex[:12],
        title=title.strip() or "(untitled)",
        text=text,
        added=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    pieces = chunk_text(text, settings.docs_chunk_chars)
    vectors = embed_passages(pieces, settings) if pieces else []

    conn = _connect(settings)
    try:
        if vectors:
            _ensure_vec_table(conn, len(vectors[0]))
        conn.execute(
            "INSERT INTO documents(id, title, text, added) VALUES (?, ?, ?, ?)",
            (doc.id, doc.title, doc.text, doc.added),
        )
        for ord_, (piece, vector) in enumerate(zip(pieces, vectors)):
            cur = conn.execute(
                "INSERT INTO chunks(doc_id, ord, text) VALUES (?, ?, ?)",
                (doc.id, ord_, piece),
            )
            conn.execute(
                f"INSERT INTO {_VEC_TABLE}(rowid, embedding) VALUES (?, ?)",
                (cur.lastrowid, _serialize(vector)),
            )
        conn.commit()
    finally:
        conn.close()
    doc.chunks = len(pieces)
    return doc


def list_documents(settings: Settings) -> list[Document]:
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.list_documents(settings)
    conn = _connect(settings)
    try:
        rows = conn.execute(
            "SELECT d.id, d.title, d.text, d.added,"
            " (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS n"
            " FROM documents d ORDER BY d.added DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        Document(id=r["id"], title=r["title"], text=r["text"], added=r["added"], chunks=r["n"])
        for r in rows
    ]


def get_document(settings: Settings, doc_id: str) -> Document | None:
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.get_document(settings, doc_id)
    conn = _connect(settings)
    try:
        row = conn.execute(
            "SELECT id, title, text, added FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return Document(id=row["id"], title=row["title"], text=row["text"], added=row["added"])


def delete_document(settings: Settings, doc_id: str) -> bool:
    """Delete a document and its chunks (and their vectors). Returns whether it existed."""
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.delete_document(settings, doc_id)
    conn = _connect(settings)
    try:
        chunk_ids = [
            r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
        ]
        existed = conn.execute(
            "SELECT 1 FROM documents WHERE id = ?", (doc_id,)
        ).fetchone() is not None
        if chunk_ids and _vec_table_exists(conn):
            conn.executemany(
                f"DELETE FROM {_VEC_TABLE} WHERE rowid = ?", [(cid,) for cid in chunk_ids]
            )
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()
    return existed


def _vec_table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM meta WHERE key = 'dim'"
    ).fetchone() is not None


def search_chunks(settings: Settings, query: str, top_k: int | None = None) -> list[Chunk]:
    """The nearest document chunks to ``query`` above ``docs_min_similarity``,
    most-similar first. Empty when nothing is ingested or nothing clears the floor."""
    query = query.strip()
    if not query:
        return []
    top_k = top_k or settings.docs_recall_top_k
    if settings.storage_backend == "postgres":
        from .. import storage_postgres

        return storage_postgres.search_chunks(settings, embed_query(query, settings), top_k)
    conn = _connect(settings)
    try:
        if not _vec_table_exists(conn):
            return []
        vector = embed_query(query, settings)
        rows = conn.execute(
            f"SELECT c.doc_id, c.text, d.title, v.distance"
            f" FROM {_VEC_TABLE} v JOIN chunks c ON c.id = v.rowid"
            f" JOIN documents d ON d.id = c.doc_id"
            f" WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (_serialize(vector), top_k),
        ).fetchall()
    finally:
        conn.close()
    results: list[Chunk] = []
    for r in rows:
        similarity = 1.0 - float(r["distance"])  # cosine distance -> similarity
        if similarity >= settings.docs_min_similarity:
            results.append(
                Chunk(doc_id=r["doc_id"], doc_title=r["title"], text=r["text"], similarity=similarity)
            )
    return results
