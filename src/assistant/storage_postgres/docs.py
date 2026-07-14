"""Document chunk tables for the Postgres backend."""

from __future__ import annotations

from datetime import UTC, datetime

from ..config import Settings
from .core import (
    _rows,
    _schema_done,
    _schema_mark,
    connect,
    vector_literal,
)


def ensure_docs_schema(settings: Settings) -> None:
    if _schema_done(settings, "docs"):
        return
    with connect(settings) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_documents (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              text TEXT NOT NULL,
              added TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_document_chunks (
              id BIGSERIAL PRIMARY KEY,
              doc_id TEXT NOT NULL REFERENCES assistant_documents(id) ON DELETE CASCADE,
              ord INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding vector NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS assistant_docs_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    _schema_mark(settings, "docs")


def docs_meta_get(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM assistant_docs_meta WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def docs_meta_set(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO assistant_docs_meta(key, value) VALUES (%s, %s) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )



def reindex_docs(settings: Settings) -> int:
    """Rebuild document chunk vectors from the stored text when the model changed.

    The docs mirror of :func:`reindex_memory`. ``assistant_document_chunks.embedding``
    is an undimensioned ``vector``, so a swapped embedding model does not fail on
    insert — it quietly mixes dimensions, and the ``<=>`` cosine operator then raises
    at query time. Rebuilding from ``assistant_documents.text`` (this store's source
    of truth) keeps every chunk on one model. No-op when the model is unchanged.
    """
    from ..docs.store import chunk_text
    from ..memory.embeddings import embed_passages

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        stored_model = docs_meta_get(conn, "embedding_model")
        rows = _rows(conn.execute("SELECT id, text FROM assistant_documents"))
        model_changed = stored_model != settings.embedding_model
        if not rows or not model_changed:
            return len(rows)
        conn.execute("DELETE FROM assistant_document_chunks")
        conn.execute(
            "DELETE FROM assistant_docs_meta WHERE key IN ('dim', 'embedding_model')"
        )

    for row in rows:
        pieces = chunk_text(str(row["text"]), settings.docs_chunk_chars)
        vectors = embed_passages(pieces, settings) if pieces else []
        if len(vectors) != len(pieces):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(pieces)} chunks"
            )
        with connect(settings) as conn:
            if vectors and docs_meta_get(conn, "dim") is None:
                docs_meta_set(conn, "dim", str(len(vectors[0])))
                docs_meta_set(conn, "embedding_model", settings.embedding_model)
            for ord_, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
                conn.execute(
                    "INSERT INTO assistant_document_chunks(doc_id, ord, text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (str(row["id"]), ord_, piece, vector_literal(vector)),
                )

    # A corpus whose documents are all empty never records a dim above; stamp the
    # model anyway so the next startup doesn't see a "changed" model and rebuild.
    with connect(settings) as conn:
        docs_meta_set(conn, "embedding_model", settings.embedding_model)
    return len(rows)


def add_document(settings: Settings, title: str, text: str, pieces: list[str], vectors: list[list[float]]):
    import uuid

    from ..docs.store import Document

    ensure_docs_schema(settings)
    doc = Document(
        id=uuid.uuid4().hex[:12],
        title=title.strip() or "(untitled)",
        text=text,
        added=datetime.now(UTC).isoformat(timespec="seconds"),
        chunks=len(pieces),
    )
    with connect(settings) as conn:
        if vectors and docs_meta_get(conn, "dim") is None:
            docs_meta_set(conn, "dim", str(len(vectors[0])))
            docs_meta_set(conn, "embedding_model", settings.embedding_model)
        conn.execute(
            "INSERT INTO assistant_documents(id, title, text, added) VALUES (%s, %s, %s, %s)",
            (doc.id, doc.title, doc.text, doc.added),
        )
        for ord_, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
            conn.execute(
                "INSERT INTO assistant_document_chunks(doc_id, ord, text, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                (doc.id, ord_, piece, vector_literal(vector)),
            )
    return doc


def list_documents(settings: Settings):
    from ..docs.store import Document

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT d.id, d.title, d.text, d.added, COUNT(c.id) AS chunks "
            "FROM assistant_documents d "
            "LEFT JOIN assistant_document_chunks c ON c.doc_id = d.id "
            "GROUP BY d.id, d.title, d.text, d.added "
            "ORDER BY d.added DESC"
        )
        rows = _rows(cur)
    return [
        Document(
            id=str(r["id"]), title=str(r["title"]), text=str(r["text"]),
            added=str(r["added"]), chunks=int(r["chunks"] or 0),
        )
        for r in rows
    ]


def get_document(settings: Settings, doc_id: str):
    from ..docs.store import Document

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "SELECT id, title, text, added FROM assistant_documents WHERE id = %s", (doc_id,)
        )
        rows = _rows(cur)
    if not rows:
        return None
    r = rows[0]
    return Document(id=str(r["id"]), title=str(r["title"]), text=str(r["text"]), added=str(r["added"]))


def delete_document(settings: Settings, doc_id: str) -> bool:
    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "DELETE FROM assistant_documents WHERE id = %s RETURNING id", (doc_id,)
        )
        return cur.fetchone() is not None


def search_chunks(settings: Settings, query_vector: list[float], top_k: int):
    from ..docs.store import Chunk

    ensure_docs_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            """
            SELECT c.doc_id, c.text, d.title,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM assistant_document_chunks c
            JOIN assistant_documents d ON d.id = c.doc_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(query_vector), vector_literal(query_vector), int(top_k)),
        )
        rows = _rows(cur)
    return [
        Chunk(
            doc_id=str(r["doc_id"]), doc_title=str(r["title"]),
            text=str(r["text"]), similarity=float(r["similarity"]),
        )
        for r in rows
        if float(r["similarity"]) >= settings.docs_min_similarity
    ]


def reindex_documents(settings: Settings, chunker, embedder) -> int:
    ensure_docs_schema(settings)
    with connect(settings) as conn:
        stored_model = docs_meta_get(conn, "embedding_model")
        documents = _rows(conn.execute("SELECT id, text FROM assistant_documents"))
        if not documents or stored_model == settings.embedding_model:
            return len(documents)
        conn.execute("DELETE FROM assistant_document_chunks")
        conn.execute("DELETE FROM assistant_docs_meta WHERE key IN ('dim', 'embedding_model')")

    for doc in documents:
        pieces = chunker(str(doc["text"]), settings.docs_chunk_chars)
        vectors = embedder(pieces, settings) if pieces else []
        if len(vectors) != len(pieces):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(pieces)} chunks"
            )
        with connect(settings) as conn:
            if vectors and docs_meta_get(conn, "dim") is None:
                docs_meta_set(conn, "dim", str(len(vectors[0])))
                docs_meta_set(conn, "embedding_model", settings.embedding_model)
            for ord_, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
                conn.execute(
                    "INSERT INTO assistant_document_chunks(doc_id, ord, text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (str(doc["id"]), ord_, piece, vector_literal(vector)),
                )
    return len(documents)
