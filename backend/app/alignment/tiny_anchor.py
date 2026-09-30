"""Timing anchors for text-only remote models.

When a remote model returns no timestamps at all, forced alignment has nothing
to anchor to. A quick local Whisper *tiny* pass provides real speech windows
and word times; the remote transcript tokens are then matched to those anchor
words (difflib), and unmatched tokens are interpolated between real anchors.
The transcript text itself stays 100% from the remote model.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field

from app.models.result import AlignedWord, RawSegment

log = logging.getLogger(__name__)

MIN_MATCH_RATIO = 0.3
_PUNCT = re.compile(r"[^\w']+", re.UNICODE)


@dataclass
class AnchorMapping:
    words: list[AlignedWord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    matched_ratio: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.words)


def normalize_token(text: str) -> str:
    return _PUNCT.sub("", text.casefold()).strip()


def _median(values: list[float], default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def map_remote_tokens(
    remote_text: str,
    anchor_words: list[AlignedWord],
    media_duration: float | None = None,
) -> AnchorMapping:
    tokens = [token for token in remote_text.split() if token.strip()]
    if not tokens:
        return AnchorMapping(warnings=["The remote transcript is empty."])
    if not anchor_words:
        return AnchorMapping(warnings=["No anchor word timings available."])

    anchor_norm = [normalize_token(word.text) for word in anchor_words]
    token_norm = [normalize_token(token) for token in tokens]

    times: list[tuple[float, float] | None] = [None] * len(tokens)
    segment_ids: list[int | None] = [None] * len(tokens)
    matched = 0
    matcher = difflib.SequenceMatcher(a=anchor_norm, b=token_norm, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            token_index = block.b + offset
            anchor = anchor_words[block.a + offset]
            times[token_index] = (anchor.start, anchor.end)
            segment_ids[token_index] = anchor.segment_id
            matched += 1

    ratio = matched / len(tokens)
    if ratio < MIN_MATCH_RATIO:
        return AnchorMapping(
            matched_ratio=ratio,
            warnings=[
                "The local anchor pass and the remote transcript did not match "
                f"({ratio * 100:.0f}% of words); subtitles were not generated."
            ],
        )

    durations = [word.end - word.start for word in anchor_words if word.end > word.start]
    fallback_duration = _median(durations, 0.35)

    # Interpolate unmatched tokens between surrounding anchors, proportionally
    # to their character length. This only spans stretches already bounded by
    # real anchor timings; nothing is invented beyond those bounds.
    index = 0
    while index < len(tokens):
        if times[index] is not None:
            index += 1
            continue
        run_start = index
        while index < len(tokens) and times[index] is None:
            index += 1
        run_end = index  # exclusive
        run = range(run_start, run_end)

        previous = times[run_start - 1] if run_start > 0 else None
        following = times[run_end] if run_end < len(tokens) else None

        if previous and following:
            span_start, span_end = previous[1], following[0]
            weights = [max(len(tokens[i]), 1) for i in run]
            total_weight = sum(weights)
            cursor = span_start
            for i, weight in zip(run, weights, strict=True):
                share = (span_end - span_start) * (weight / total_weight)
                times[i] = (cursor, cursor + max(share, 0.05))
                cursor += share
        elif previous:
            cursor = previous[1]
            for i in run:
                times[i] = (cursor, cursor + fallback_duration)
                cursor += fallback_duration
        elif following:
            cursor = following[0]
            for i in reversed(list(run)):
                start = max(0.0, cursor - fallback_duration)
                times[i] = (start, cursor)
                cursor = start
        else:  # pragma: no cover - cannot happen when ratio >= MIN_MATCH_RATIO
            cursor = 0.0
            for i in run:
                times[i] = (cursor, cursor + fallback_duration)
                cursor += fallback_duration

    words: list[AlignedWord] = []
    for i, token in enumerate(tokens):
        start, end = times[i] or (0.0, fallback_duration)
        if media_duration is not None and start >= media_duration:
            continue
        if media_duration is not None:
            end = min(end, media_duration)
        words.append(
            AlignedWord(
                text=token,
                start=round(start, 3),
                end=round(max(end, start + 0.01), 3),
                probability=None,
                segment_id=segment_ids[i],
            )
        )

    return AnchorMapping(
        words=words,
        matched_ratio=ratio,
        warnings=[],
    )


def build_alignment_windows(
    mapped_words: list[AlignedWord],
    anchor_segments: list[RawSegment],
) -> list[dict]:
    """Group mapped remote tokens into the anchor segment windows for WhisperX."""
    by_id: dict[int, RawSegment] = {segment.id: segment for segment in anchor_segments}
    groups: dict[int | None, list[AlignedWord]] = {}
    last_known: int | None = None
    for word in mapped_words:
        segment_id = word.segment_id
        if segment_id is None:
            segment_id = last_known
            word = word.model_copy(update={"segment_id": segment_id})
        else:
            last_known = segment_id
        groups.setdefault(segment_id, []).append(word)

    windows: list[dict] = []
    for segment_id, words in groups.items():
        segment = by_id.get(segment_id) if segment_id is not None else None
        if segment is None or not words:
            continue
        text = " ".join(word.text for word in words)
        windows.append({"start": segment.start, "end": segment.end, "text": text})
    windows.sort(key=lambda window: window["start"])
    return windows
