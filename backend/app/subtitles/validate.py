"""Cue validation and conservative repair.

Runs after segmentation and before serialization. Cues are checked for
finite/non-negative timestamps, ordering, overlap, media-duration bounds and
non-empty text. Quantized millisecond values (used by SRT/VTT) are validated
separately so serialized subtitles can never overlap by rounding.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from app.models.result import AlignedWord, Cue
from app.subtitles.segmenter import _wrap_words  # rewrap after repair

log = logging.getLogger(__name__)

_WORD_RANGE_TOLERANCE = 0.75


@dataclass
class ValidationResult:
    cues: list[Cue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class QuantizedCue:
    index: int
    start_ms: int
    end_ms: int
    text: str


def validate_and_repair_cues(
    cues: list[Cue],
    words: list[AlignedWord],
    media_duration: float | None = None,
    max_line_chars: int = 42,
    max_lines: int = 2,
) -> ValidationResult:
    warnings: list[str] = []
    kept: list[Cue] = []

    ordered = sorted(cues, key=lambda c: (c.start, c.end))

    for cue in ordered:
        start, end = float(cue.start), float(cue.end)
        text = cue.text.strip()
        if not text:
            warnings.append(f"Dropped cue {cue.index}: empty text.")
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            warnings.append(f"Dropped cue {cue.index}: non-finite timestamp.")
            continue
        if start < 0:
            warnings.append(f"Cue {cue.index}: negative start clamped to 0.")
            start = 0.0
        if media_duration is not None and end > media_duration + 1e-6:
            warnings.append(
                f"Cue {cue.index}: end beyond media duration clamped to {media_duration:.3f}s."
            )
            end = media_duration
        if end - start <= 0.02:
            warnings.append(f"Dropped cue {cue.index}: start is not before end.")
            continue

        candidate = cue.model_copy(
            update={
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "lines": text.splitlines(),
            }
        )

        if kept and candidate.start < kept[-1].end - 1e-6:
            previous = kept[-1]
            if candidate.start <= previous.start + 0.02:
                # Fully overlapped or out of order: merge the texts.
                merged_start = min(previous.start, candidate.start)
                merged_end = max(previous.end, candidate.end)
                merged_words = words[
                    min(previous.first_word_index, candidate.first_word_index) : max(
                        previous.last_word_index, candidate.last_word_index
                    )
                ]
                lines = _wrap_words(merged_words, max_line_chars)[:max_lines] if merged_words else []
                merged_text = "\n".join(lines) if lines else f"{previous.text} {candidate.text}"
                warnings.append(
                    f"Merged cues {previous.index} and {candidate.index} after an overlap."
                )
                kept[-1] = previous.model_copy(
                    update={
                        "start": round(merged_start, 3),
                        "end": round(merged_end, 3),
                        "lines": lines or [merged_text],
                        "text": merged_text,
                        "first_word_index": min(
                            previous.first_word_index, candidate.first_word_index
                        ),
                        "last_word_index": max(
                            previous.last_word_index, candidate.last_word_index
                        ),
                    }
                )
                continue
            trimmed = round(min(previous.end, candidate.start - 0.001), 3)
            if trimmed > previous.start + 0.02:
                warnings.append(
                    f"Trimmed cue {previous.index} end to avoid overlapping cue {candidate.index}."
                )
                kept[-1] = previous.model_copy(
                    update={
                        "end": trimmed,
                        "text": "\n".join(previous.lines),
                    }
                )
            else:
                warnings.append(
                    f"Dropped cue {previous.index}: could not resolve overlap with cue {candidate.index}."
                )
                kept.pop()
        kept.append(candidate)

    for position, cue in enumerate(kept, start=1):
        cue.index = position
        expected_words = words[cue.first_word_index : cue.last_word_index]
        if expected_words:
            first, last = expected_words[0], expected_words[-1]
            if (
                first.start < cue.start - _WORD_RANGE_TOLERANCE
                or last.end > cue.end + _WORD_RANGE_TOLERANCE
            ):
                warnings.append(
                    f"Cue {cue.index}: word timestamps fall outside the cue range."
                )
        cue.text = "\n".join(cue.lines) if cue.lines else cue.text

    return ValidationResult(cues=kept, warnings=warnings)


def quantize_cues(cues: list[Cue]) -> tuple[list[QuantizedCue], list[str]]:
    """Round to milliseconds and guarantee non-overlapping serialized values."""
    warnings: list[str] = []
    quantized: list[QuantizedCue] = []
    for cue in cues:
        start_ms = max(0, int(round(cue.start * 1000)))
        end_ms = int(round(cue.end * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        quantized.append(
            QuantizedCue(
                index=len(quantized) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                text=cue.text,
            )
        )
    index = 1
    while index < len(quantized):
        previous, current = quantized[index - 1], quantized[index]
        if current.start_ms < previous.end_ms:
            new_end = current.start_ms - 1
            if new_end - previous.start_ms >= 10:
                warnings.append(
                    f"Serialized cue {previous.index} shortened by rounding to avoid overlap."
                )
                previous.end_ms = new_end
            else:
                warnings.append(
                    f"Serialized cue {previous.index} dropped: rounding could not resolve overlap."
                )
                quantized.pop(index - 1)
                index = max(1, index - 1)
                continue
        index += 1
    for position, cue in enumerate(quantized, start=1):
        cue.index = position
    return quantized, warnings
