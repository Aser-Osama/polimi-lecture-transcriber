"""Plain transcript serializer."""

from __future__ import annotations

from app.models.result import RawSegment
from app.subtitles.textfmt import build_plain_text


def render(segments: list[RawSegment]) -> str:
    text = build_plain_text(segments)
    return text + "\n" if text else ""
