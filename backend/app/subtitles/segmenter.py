"""Word-timestamp based subtitle segmentation.

Every cue is built from the real start/end timestamps of the words it
contains. Timestamps are never fabricated or evenly distributed.

Break rules, in priority order:
1. a meaningful silence between words (>= silence_break)
2. sentence-final punctuation when the cue is already readable
3. weak punctuation (; :) with a pause
4. the maximum cue duration
5. the character budget (max_line_chars * max_lines)

Afterwards small cues are merged when possible, cue ends get a small padding
bounded by the next cue's start, and any cue that would still wrap to more
than ``max_lines`` lines (for example a single very long whisper segment) is
split at a word boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.models.result import AlignedWord, Cue, RawSegment

_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
_WEAK_END = (";", ":", "，", ",")
_CLOSERS = "\"'\u201d\u2019)]}"
_MIN_WORD_DURATION = 0.01
_EPS = 1e-9  # float comparison tolerance for time gaps


@dataclass
class SegmenterOptions:
    max_line_chars: int = 42
    max_lines: int = 2
    max_cue_duration: float = 7.0
    min_cue_duration: float = 0.7
    silence_break: float = 0.6
    sentence_pause: float = 0.15
    sentence_min_duration: float = 1.0
    weak_break_min_duration: float = 2.0
    weak_break_pause: float = 0.25
    end_padding: float = 0.3
    max_gap_merge: float = 0.35
    merge_duration_slack: float = 1.25
    media_tail_tolerance: float = 0.5

    @property
    def char_budget(self) -> int:
        return self.max_line_chars * self.max_lines


@dataclass
class SegmenterResult:
    cues: list[Cue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def segment_words(
    words: list[AlignedWord],
    options: SegmenterOptions | None = None,
    media_duration: float | None = None,
) -> SegmenterResult:
    opts = options or SegmenterOptions()
    warnings: list[str] = []
    clean = _sanitize_words(words, opts, media_duration, warnings)
    if not clean:
        if words:
            warnings.append("No usable word timestamps; subtitles were not generated.")
        return SegmenterResult(cues=[], warnings=warnings)

    groups = _build_groups(clean, opts)
    groups = _merge_tiny_groups(groups, opts, warnings)
    groups = _split_overflow_groups(groups, opts, warnings)
    cues = _finalize(clean, groups, opts, media_duration, warnings)
    return SegmenterResult(cues=cues, warnings=warnings)


def segment_from_segments(
    segments: list[RawSegment],
    options: SegmenterOptions | None = None,
    media_duration: float | None = None,
) -> SegmenterResult:
    """Fallback when no word timestamps exist: group raw whisper segments.

    Segment boundaries are used as-is (never fabricated word timings); groups
    are limited by duration and character budget so cues remain readable.
    """
    opts = options or SegmenterOptions()
    warnings = [
        "Subtitles were built from raw Whisper segment timing because "
        "word-level timestamps were unavailable. Timing is less precise."
    ]
    usable = [s for s in segments if s.text.strip() and s.end > s.start]
    if not usable:
        return SegmenterResult(cues=[], warnings=warnings)

    groups: list[list[RawSegment]] = []
    current: list[RawSegment] = []
    for segment in usable:
        if current:
            duration = segment.end - current[0].start
            chars = len(" ".join(s.text.strip() for s in current)) + 1 + len(segment.text.strip())
            gap = segment.start - current[-1].end
            if duration > opts.max_cue_duration or chars > opts.char_budget or gap >= opts.silence_break:
                groups.append(current)
                current = []
        current.append(segment)
    if current:
        groups.append(current)

    cues: list[Cue] = []
    for index, group in enumerate(groups, start=1):
        text = " ".join(s.text.strip() for s in group)
        words = [w for w in text.split() if w]
        lines: list[str] = []
        line = ""
        for token in words:
            candidate = f"{line} {token}".strip()
            if line and len(candidate) > opts.max_line_chars:
                lines.append(line)
                line = token
            else:
                line = candidate
        if line:
            lines.append(line)
        if len(lines) > opts.max_lines:
            lines = lines[: opts.max_lines]
            warnings.append(f"Cue {index} exceeded the line limit; text was truncated.")
        start = group[0].start
        end = group[-1].end
        if media_duration is not None:
            end = min(end, media_duration)
        cues.append(
            Cue(
                index=index,
                start=round(max(0.0, start), 3),
                end=round(end, 3),
                lines=lines,
                text="\n".join(lines),
                first_word_index=0,
                last_word_index=0,
            )
        )
    return SegmenterResult(cues=cues, warnings=warnings)


def _sanitize_words(
    words: list[AlignedWord],
    opts: SegmenterOptions,
    media_duration: float | None,
    warnings: list[str],
) -> list[AlignedWord]:
    clean: list[AlignedWord] = []
    dropped = 0
    clamped = 0
    prev_end = 0.0
    for word in words:
        text = word.text.strip()
        if not text:
            dropped += 1
            continue
        start, end = float(word.start), float(word.end)
        if not math.isfinite(start) or not math.isfinite(end):
            dropped += 1
            continue
        if start < 0:
            start = 0.0
            clamped += 1
        if media_duration is not None:
            if start >= media_duration:
                dropped += 1
                continue
            if end > media_duration:
                end = media_duration
                clamped += 1
        if end - start < _MIN_WORD_DURATION:
            if start < end:
                end = start + _MIN_WORD_DURATION
            else:
                dropped += 1
                continue
        if start < prev_end:
            start = prev_end
            clamped += 1
            if start >= end - _MIN_WORD_DURATION:
                end = start + _MIN_WORD_DURATION
        if start >= end:
            dropped += 1
            continue
        clean.append(
            AlignedWord(
                text=text,
                start=round(start, 3),
                end=round(end, 3),
                probability=word.probability,
                segment_id=word.segment_id,
            )
        )
        prev_end = end
    if dropped:
        warnings.append(f"Ignored {dropped} malformed word timestamp(s).")
    if clamped:
        warnings.append(f"Repaired {clamped} out-of-range word timestamp(s).")
    return clean


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip(_CLOSERS)
    return stripped.endswith(_SENTENCE_END)


def _ends_weak(text: str) -> bool:
    stripped = text.rstrip(_CLOSERS)
    return stripped.endswith(_WEAK_END)


def _join(words: list[AlignedWord]) -> str:
    return " ".join(w.text for w in words)


def _duration(words: list[AlignedWord]) -> float:
    return words[-1].end - words[0].start


def _build_groups(
    words: list[AlignedWord], opts: SegmenterOptions
) -> list[list[AlignedWord]]:
    groups: list[list[AlignedWord]] = []
    current: list[AlignedWord] = []
    for word in words:
        if current and _should_break(current, word, opts):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def _should_break(
    current: list[AlignedWord], word: AlignedWord, opts: SegmenterOptions
) -> bool:
    prev = current[-1]
    gap = word.start - prev.end
    duration = _duration(current)

    if gap >= opts.silence_break - _EPS:
        return True

    if (
        _ends_sentence(prev.text)
        and duration >= opts.sentence_min_duration - _EPS
        and gap >= opts.sentence_pause - _EPS
    ):
        return True

    if (
        _ends_weak(prev.text)
        and duration >= opts.weak_break_min_duration - _EPS
        and gap >= opts.weak_break_pause - _EPS
    ):
        return True

    if word.end - current[0].start > opts.max_cue_duration:
        return True

    projected = len(_join(current)) + 1 + len(word.text)
    return projected > opts.char_budget


def _merge_groups(
    left: list[AlignedWord], right: list[AlignedWord]
) -> list[AlignedWord]:
    return left + right


def _can_merge(
    left: list[AlignedWord], right: list[AlignedWord], opts: SegmenterOptions
) -> bool:
    gap = right[0].start - left[-1].end
    if gap > opts.max_gap_merge:
        return False
    combined_duration = right[-1].end - left[0].start
    if combined_duration > opts.max_cue_duration * opts.merge_duration_slack:
        return False
    combined_chars = len(_join(left)) + 1 + len(_join(right))
    return combined_chars <= opts.char_budget + 12


def _merge_tiny_groups(
    groups: list[list[AlignedWord]], opts: SegmenterOptions, warnings: list[str]
) -> list[list[AlignedWord]]:
    merged: list[list[AlignedWord]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        if _duration(group) < opts.min_cue_duration:
            if merged and _can_merge(merged[-1], group, opts):
                merged[-1] = _merge_groups(merged[-1], group)
                index += 1
                continue
            if index + 1 < len(groups) and _can_merge(group, groups[index + 1], opts):
                merged.append(_merge_groups(group, groups[index + 1]))
                index += 2
                continue
        merged.append(group)
        index += 1
    return merged


def _wrap_words(words: list[AlignedWord], max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word.text}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word.text
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _split_overflow_groups(
    groups: list[list[AlignedWord]], opts: SegmenterOptions, warnings: list[str]
) -> list[list[AlignedWord]]:
    result: list[list[AlignedWord]] = []
    for group in groups:
        result.extend(_split_group(group, opts, warnings))
    return result


def _split_group(
    group: list[AlignedWord], opts: SegmenterOptions, warnings: list[str]
) -> list[list[AlignedWord]]:
    if len(_wrap_words(group, opts.max_line_chars)) <= opts.max_lines or len(group) < 2:
        return [group]
    best: int | None = None
    best_score = None
    middle = len(group) / 2
    for k in range(1, len(group)):
        left_ok = len(_wrap_words(group[:k], opts.max_line_chars)) <= opts.max_lines
        right_ok = (
            len(_wrap_words(group[k:], opts.max_line_chars)) <= opts.max_lines
        )
        if left_ok and right_ok:
            score = abs(k - middle)
            if best_score is None or score < best_score:
                best = k
                best_score = score
    if best is None:
        best = max(1, len(group) // 2)
        warnings.append("Split an over-long subtitle cue at a near word boundary.")
    left = _split_group(group[:best], opts, warnings)
    right = _split_group(group[best:], opts, warnings)
    return left + right


def _finalize(
    words: list[AlignedWord],
    groups: list[list[AlignedWord]],
    opts: SegmenterOptions,
    media_duration: float | None,
    warnings: list[str],
) -> list[Cue]:
    word_positions = {id(word): index for index, word in enumerate(words)}
    seen: set[int] = set()
    spans: list[tuple[int, int]] = []
    for group in groups:
        indices = []
        for word in group:
            key = word_positions.get(id(word))
            if key is None:
                continue
            indices.append(key)
        if not indices:
            continue
        first, last = min(indices), max(indices)
        spans.append((first, last + 1))
        seen.update(indices)
    spans.sort()

    # Pad ends, never overlapping the next cue or the media duration.
    padded: list[tuple[float, float]] = []
    for position, (first, last) in enumerate(spans):
        start = words[first].start
        end = words[last - 1].end
        limit = media_duration if media_duration is not None else float("inf")
        if position + 1 < len(spans):
            next_start = words[spans[position + 1][0]].start
            limit = min(limit, next_start - 0.05)
        padded_end = min(end + opts.end_padding, limit)
        if padded_end > end:
            end = padded_end
        padded.append((start, max(start + _MIN_WORD_DURATION, end)))

    cues: list[Cue] = []
    short_flagged = 0
    for index, ((first, last), (start, end)) in enumerate(zip(spans, padded, strict=True)):
        cue_words = words[first:last]
        lines = _wrap_words(cue_words, opts.max_line_chars)
        if len(lines) > opts.max_lines:
            lines = lines[: opts.max_lines]
            warnings.append(f"Cue {index + 1} exceeded the line limit; text was truncated.")
        cues.append(
            Cue(
                index=index + 1,
                start=round(start, 3),
                end=round(end, 3),
                lines=lines,
                text="\n".join(lines),
                first_word_index=first,
                last_word_index=last,
            )
        )
        if end - start < opts.min_cue_duration:
            short_flagged += 1
    if short_flagged:
        warnings.append(
            f"{short_flagged} cue(s) are shorter than the preferred minimum "
            f"({opts.min_cue_duration:.1f}s) because of surrounding silences."
        )
    return cues
