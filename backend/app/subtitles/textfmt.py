"""Plain-text transcript building.

Conservative cleanup only: whitespace normalization, punctuation spacing and
paragraph breaks at long pauses. The recognized content is never rewritten.
"""

from __future__ import annotations

import re

from app.models.result import RawSegment

_MULTI_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")
_PARAGRAPH_GAP = 2.0


def normalize_sentence(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text.replace("\n", " "))
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return text.strip()


def build_plain_text(segments: list[RawSegment]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    previous_end: float | None = None
    for segment in segments:
        text = normalize_sentence(segment.text)
        if not text:
            continue
        if (
            previous_end is not None
            and segment.start - previous_end >= _PARAGRAPH_GAP
            and current
        ):
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        previous_end = segment.end
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()
