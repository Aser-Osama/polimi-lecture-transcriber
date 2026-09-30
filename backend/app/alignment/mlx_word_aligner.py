"""Aligner for MLX Whisper's cross-attention word timestamps.

The mlx-whisper provider already returns DTW-aligned word timings when
``word_timestamps=True`` (word/start/end/probability). This aligner preserves
those timings, flattens them out of the raw whisper segments, removes
non-finite values and reports what it had to adjust. It never invents times.
"""

from __future__ import annotations

import math

from app.alignment.base import AlignmentProvider, AlignmentResult, NOOP_PROGRESS
from app.models.result import AlignedWord
from app.providers.base import CancellationToken, ProgressCallback, RawTranscription


class MLXWordTimestampAligner(AlignmentProvider):
    name = "mlx_word_timestamps"

    def align(
        self,
        transcription: RawTranscription,
        media_duration: float | None = None,
        progress: ProgressCallback = NOOP_PROGRESS,
        cancel: CancellationToken | None = None,
    ) -> AlignmentResult:
        if cancel is not None:
            cancel.raise_if_cancelled()
        progress("aligning", None)

        warnings: list[str] = []
        words = []
        dropped = 0
        for segment in transcription.segments:
            for word in segment.words:
                start, end = float(word.start), float(word.end)
                if not math.isfinite(start) or not math.isfinite(end):
                    dropped += 1
                    continue
                if end < start:
                    start, end = end, start
                text = word.text.strip()
                if not text:
                    dropped += 1
                    continue
                if media_duration is not None and start >= media_duration:
                    dropped += 1
                    continue
                words.append(
                    {
                        "text": text,
                        "start": start,
                        "end": end,
                        "probability": word.probability,
                        "segment_id": segment.id,
                    }
                )

        if dropped:
            warnings.append(f"Alignment dropped {dropped} unusable word timestamp(s).")

        words.sort(key=lambda w: (w["start"], w["end"]))
        aligned = [AlignedWord(**w) for w in words]

        if not aligned:
            warnings.append(
                "Whisper returned no word-level timestamps for this media; "
                "subtitles fall back to raw segment timing."
            )
        return AlignmentResult(words=aligned, warnings=warnings)
