"""Document extraction tests — PDF/DOCX/HTML to text, and ingest-then-search.

The PDF is handcrafted (offsets computed, so the xref is valid) and the DOCX is
generated with python-docx, so both real parsers run — no fixtures on disk, no
network. Embeddings are faked as in test_docs.py. Ingestion is driven through
the store directly (the daemon has no HTTP surface); the model reaches these
paths through its document tools.
"""

from __future__ import annotations

import io
import math
import re
import zlib

import pytest

from assistant.config import Settings
from assistant.docs import extract
from assistant.docs import store as docs_store


def _tiny_pdf(text: str) -> bytes:
    """A minimal one-page PDF containing ``text``, with a correct xref table."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_at = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF".encode()
    )
    return out.getvalue()


def _tiny_docx(paragraphs: list[str]) -> bytes:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_extract_pdf() -> None:
    text = extract.extract_text("report.pdf", _tiny_pdf("Hello PDF"))
    assert "Hello PDF" in text


def test_extract_pdf_by_magic_bytes_without_extension() -> None:
    assert "Hello PDF" in extract.extract_text("upload.bin", _tiny_pdf("Hello PDF"))


def test_extract_textless_pdf_raises() -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(extract.ExtractionError, match="no extractable text"):
        extract.extract_text("blank.pdf", buf.getvalue())


def test_extract_docx() -> None:
    content = _tiny_docx(["First paragraph.", "Second paragraph."])
    text = extract.extract_text("notes.docx", content)
    assert "First paragraph." in text and "Second paragraph." in text


def test_extract_corrupt_docx_raises() -> None:
    with pytest.raises(extract.ExtractionError):
        extract.extract_text("broken.docx", b"not a zip at all")


def test_pdf_extraction_wraps_non_pypdf_errors(monkeypatch) -> None:
    # pypdf can raise assorted non-PyPdfError types deep in extract_text() on
    # adversarial PDFs; the boundary must still fail as ExtractionError (a clean
    # 422 upstream), never let a raw exception escape to a 500.
    import pypdf

    class _BoomReader:
        def __init__(self, *a, **k):
            pass

        @property
        def pages(self):
            raise KeyError("/Contents")  # not a PyPdfError

    monkeypatch.setattr(pypdf, "PdfReader", _BoomReader)
    with pytest.raises(extract.ExtractionError, match="could not read PDF"):
        extract.extract_text("weird.pdf", b"%PDF-1.4 ...")


def test_extract_plain_text_decodes() -> None:
    assert extract.extract_text("plan.txt", "hei på deg".encode()) == "hei på deg"


def test_html_to_text_drops_scripts_and_keeps_prose() -> None:
    html = (
        "<html><head><title>T</title><style>p{color:red}</style></head>"
        "<body><script>alert(1)</script><h1>Heading</h1><p>Some prose.</p></body></html>"
    )
    text = extract.html_to_text(html)
    assert "Heading" in text and "Some prose." in text
    assert "alert" not in text and "color" not in text


def test_fetch_url_rejects_non_http() -> None:
    with pytest.raises(extract.ExtractionError, match="http"):
        extract.fetch_url_text("file:///etc/passwd")


# --- ingest endpoints --------------------------------------------------------- #


def _fake_embed(texts, prefix: str = "", settings=None):
    vecs = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setattr("assistant.memory.embeddings._embed", _fake_embed)
    return Settings(memory_dir=str(tmp_path / "memory"), docs_min_similarity=0.1)


def test_pdf_extracts_ingests_and_is_searchable(settings) -> None:
    # The pipeline a document tool drives: raw bytes → extracted text →
    # chunked/embedded/stored → recalled by semantic search.
    text = extract.extract_text("hello.pdf", _tiny_pdf("The fjord tour starts in Flam"))
    doc = docs_store.add_document(settings, "hello.pdf", text)
    assert doc.title == "hello.pdf" and doc.chunks >= 1

    found = docs_store.search_chunks(settings, "fjord tour Flam")
    assert found and "fjord" in found[0].text


def test_unreadable_upload_raises_extraction_error(settings) -> None:
    with pytest.raises(extract.ExtractionError):
        extract.extract_text("broken.docx", b"junk")
