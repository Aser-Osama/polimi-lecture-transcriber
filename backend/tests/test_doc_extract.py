from __future__ import annotations

import pytest

from app.services.doc_extract import (
    DocumentExtractionError,
    extract_document_text,
    extract_terms,
    merge_terms_into_context,
)


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF with a correct xref table."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    body_objects = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
        ),
        (4, b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"),
        (5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in body_objects:
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_position = len(out)
    out += f"xref\n0 {len(body_objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for number, _ in body_objects:
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(body_objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_extract_terms_finds_acronyms_and_phrases():
    text = (
        "Virtual memory and the TLB are covered in this lecture by "
        "Daniele Cattaneo at Politecnico di Milano. The TLB caches translations. "
        "NUMA systems and the MESI protocol appear in the NUMA section."
    )
    terms = extract_terms(text)
    assert "TLB" in terms
    assert "NUMA" in terms
    assert "MESI" in terms
    assert "Daniele Cattaneo" in terms
    assert "Politecnico di Milano" in terms


def test_extract_terms_ignores_document_stopwords_and_sentence_words():
    text = "This PDF was exported. PDF is a format. We discuss algorithms."
    terms = extract_terms(text)
    assert "PDF" not in terms
    assert "We" not in terms


def test_extract_terms_requires_frequency_for_single_names():
    text = "Cattaneo spoke first. Later, Cattaneo returned. Finally Cattaneo left."
    terms = extract_terms(text)
    assert "Cattaneo" in terms  # frequency 3
    text_once = "Cattaneo spoke first."
    assert "Cattaneo" not in extract_terms(text_once)


def test_extract_terms_deduplicates_case_insensitively():
    terms = extract_terms("TLB tlb TLB NUMA")
    assert terms.count("TLB") == 1
    assert terms.count("tlb") == 0


def test_extract_terms_respects_cap():
    text = " ".join(f"TL{i:02d}" for i in range(300))
    assert len(extract_terms(text)) <= 120


def test_extract_text_txt_and_markdown():
    assert "NUMA" in extract_document_text("notes.txt", b"NUMA and TLB")
    assert "MESI" in extract_document_text("notes.md", b"# Lecture\nMESI protocol")


def test_extract_text_rejects_unsupported_types():
    with pytest.raises(DocumentExtractionError):
        extract_document_text("slides.docx", b"x")


def test_extract_text_rejects_empty_documents():
    with pytest.raises(DocumentExtractionError):
        extract_document_text("notes.txt", b"   \n")


def test_extract_text_pdf():
    pdf = make_pdf("TLB NUMA MESI and the Daniele Cattaneo lecture")
    text = extract_document_text("slides.pdf", pdf)
    assert "TLB" in text
    assert "Daniele Cattaneo" in text


def test_merge_terms_into_context_appends_only_new():
    context = "NUMA\nTLB"
    merged, added = merge_terms_into_context(context, ["TLB", "MESI", "Cattaneo"])
    assert added == ["MESI", "Cattaneo"]
    assert merged == "NUMA\nTLB\nMESI\nCattaneo"
    merged2, added2 = merge_terms_into_context(merged, ["TLB", "MESI"])
    assert added2 == []
    assert merged2 == merged


def test_merge_terms_into_empty_context():
    merged, added = merge_terms_into_context("", ["TLB"])
    assert merged == "TLB"
    assert added == ["TLB"]
