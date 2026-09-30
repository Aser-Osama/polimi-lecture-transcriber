"""Extract names/acronyms/terminology from PDF, TXT or Markdown documents.

Deliberately lightweight: plain-text extraction plus explainable frequency
heuristics. No LLM, no embeddings, no vector database. The extracted terms are
meant to be appended to a course/project context and edited by the user.
"""

from __future__ import annotations

import io
import re
from collections import Counter

from app.core.errors import AppError

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TERMS = 120
MAX_TEXT_CHARS = 500_000

_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][\w'’.-]*", re.UNICODE)
_ACRONYM = re.compile(r"^[A-Z][A-Z0-9]{1,7}(?:[-/][A-Z0-9]{1,6})*$")
_CONNECTORS = {
    "di", "de", "del", "della", "delle", "degli", "van", "von", "der", "den",
    "la", "le", "of", "the", "and", "und", "et", "&",
}
# All-caps words that are almost never technical terms in a lecture document.
_ACRONYM_STOP = {
    "PDF", "MD", "TXT", "URL", "HTTP", "HTTPS", "WWW", "AND", "THE", "OR", "OF",
    "TO", "IN", "IS", "FOR", "ON", "WITH", "BY", "AS", "AT", "BE", "IT", "AN",
    "IF", "OK", "ID", "NO", "YES", "ALL", "NOT", "YOU", "WE", "I", "A",
}
_SINGLE_CAP_FREQUENCY = 3
_MIN_PHRASE_WORDS = 2
_MAX_PHRASE_WORDS = 5
_SENTENCE_BOUNDARY = ".!?;:\n"
# Capitalized function/sentence-starter words: they may continue a name but
# never start one (avoids phrases like "Later Cattaneo").
_SENTENCE_STARTERS = {
    "The", "This", "That", "These", "Those", "A", "An", "We", "It", "Its",
    "In", "On", "At", "For", "And", "But", "Or", "If", "When", "While",
    "As", "By", "To", "From", "With", "Here", "There", "Today", "Next",
    "Now", "Then", "Also", "However", "So", "Thus", "Hence", "Therefore",
    "First", "Second", "Third", "Finally", "Later", "Consider", "Suppose",
    "Note", "Example", "Figure", "Table", "Our", "Their", "His", "Her",
    "Chapter", "Section", "Exercise", "Definition", "Theorem", "Proof",
}


class DocumentExtractionError(AppError):
    code = "document_extraction"
    user_message = "The document could not be read."


def extract_document_text(filename: str, data: bytes) -> str:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentExtractionError(
            f"Unsupported document type: {suffix or filename}",
            user_message="Only PDF, TXT and Markdown files are supported.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentExtractionError(
            f"Document too large: {len(data)} bytes",
            user_message="The document is larger than 20 MB.",
        )
    text = _extract_pdf_text(data) if suffix == ".pdf" else _decode_text(data)
    if not text.strip():
        raise DocumentExtractionError(
            "No extractable text found",
            user_message="No text could be extracted from this document "
            "(scanned/image-only PDFs are not supported).",
        )
    return text[:MAX_TEXT_CHARS]


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise DocumentExtractionError(
            f"pypdf is not installed: {exc}",
            user_message="PDF support is not installed. Run ./setup.sh.",
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises several error types
        raise DocumentExtractionError(
            f"PDF parsing failed: {exc}",
            user_message="This PDF could not be read. It may be corrupted or encrypted.",
        ) from exc
    return "\n".join(pages)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentExtractionError(
        "Text decoding failed",
        user_message="This text file could not be decoded as UTF-8.",
    )


def extract_terms(text: str, max_terms: int = MAX_TERMS) -> list[str]:
    """Frequency-based candidates: acronyms, capitalized phrases, repeated names."""
    tokens = [(match.group(), match.start(), match.end()) for match in _WORD.finditer(text)]
    acronym_counts: Counter[str] = Counter()
    single_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()

    def ends_sentence(token_text: str) -> bool:
        # Word tokens may swallow a trailing period ("Milano."), so the token's
        # final character counts as a boundary marker too.
        return bool(token_text) and token_text[-1] in _SENTENCE_BOUNDARY

    def sentence_starts_at(index: int) -> bool:
        if index == 0:
            return True
        previous_text, _, previous_end = tokens[index - 1]
        if ends_sentence(previous_text):
            return True
        gap = text[previous_end : tokens[index][1]]
        return any(char in gap for char in _SENTENCE_BOUNDARY)

    def crosses_sentence(previous_text: str, previous_end: int, next_start: int) -> bool:
        if ends_sentence(previous_text):
            return True
        gap = text[previous_end:next_start]
        return any(char in gap for char in _SENTENCE_BOUNDARY)

    index = 0
    while index < len(tokens):
        word, _, word_end = tokens[index]
        if _ACRONYM.match(word):
            if word.upper() not in _ACRONYM_STOP:
                acronym_counts[word] += 1
            index += 1
            continue
        if word[:1].isupper() and not word.isupper():
            can_start_phrase = not (
                sentence_starts_at(index) and word in _SENTENCE_STARTERS
            )
            phrase = [word]
            lookahead = index + 1
            previous_text, previous_end = word, word_end
            while can_start_phrase and lookahead < len(tokens) and len(phrase) < _MAX_PHRASE_WORDS:
                candidate, candidate_start, candidate_end = tokens[lookahead]
                if crosses_sentence(previous_text, previous_end, candidate_start):
                    break
                if candidate in _CONNECTORS:
                    if (
                        lookahead + 1 < len(tokens)
                        and tokens[lookahead + 1][0][:1].isupper()
                        and not crosses_sentence(
                            candidate, candidate_end, tokens[lookahead + 1][1]
                        )
                    ):
                        phrase.extend([candidate, tokens[lookahead + 1][0]])
                        previous_text, previous_end = tokens[lookahead + 1][0], tokens[lookahead + 1][2]
                        lookahead += 2
                        continue
                    break
                if candidate[:1].isupper() and not candidate.isupper():
                    phrase.append(candidate)
                    previous_text, previous_end = candidate, candidate_end
                    lookahead += 1
                    continue
                break
            if len(phrase) >= _MIN_PHRASE_WORDS:
                phrase_counts[" ".join(phrase)] += 1
                index = lookahead
                continue
            if not sentence_starts_at(index) or word not in _SENTENCE_STARTERS:
                single_counts[word] += 1
            index += 1
            continue
        index += 1

    ranked: list[tuple[int, str]] = []
    for term, count in acronym_counts.items():
        ranked.append((count + 10, term))  # acronyms are strong signals
    for term, count in phrase_counts.items():
        ranked.append((count + 5, term))  # multiword proper nouns
    for term, count in single_counts.items():
        if count >= _SINGLE_CAP_FREQUENCY:
            ranked.append((count, term))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    seen: set[str] = set()
    terms: list[str] = []
    for _, term in ranked:
        term = term.strip(".,;:!?()[]{}").strip()
        if not term:
            continue
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def merge_terms_into_context(context: str, terms: list[str]) -> tuple[str, list[str]]:
    """Append terms to a context text, skipping ones already present."""
    from app.services.glossary import clean_glossary_terms

    existing = {term.casefold() for term in clean_glossary_terms(context)}
    added: list[str] = []
    for term in terms:
        if term.casefold() in existing:
            continue
        existing.add(term.casefold())
        added.append(term)
    if not added:
        return context.rstrip(), []
    merged = context.rstrip()
    if merged:
        merged += "\n"
    merged += "\n".join(added)
    return merged, added
