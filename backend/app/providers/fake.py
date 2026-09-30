"""Deterministic fake provider.

Used by unit tests and by UI development (PT_PROVIDER=fake) so no model needs
to be downloaded or run. It honors cancellation between segments.
"""

from __future__ import annotations

import time

from app.core.errors import TranscriptionError
from app.models.result import RawSegment, RawWord
from app.providers.base import (
    CancellationToken,
    ProgressCallback,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
)

_SCRIPT = [
    "Good morning everyone and welcome to the operating systems lecture.",
    "Today we will discuss virtual memory and the page table structure.",
    "The TLB caches recent virtual to physical address translations.",
    "Cache coherence in multicore systems relies on the MESI protocol.",
    "A spinlock busy waits while a mutex can sleep the current thread.",
    "CUDA kernels execute on the GPU and SIMD units process vectors.",
    "Remember that the hypervisor multiplexes the physical hardware.",
    "That concludes the lecture, please review the microarchitecture notes.",
]


class FakeProvider(TranscriptionProvider):
    name = "fake"

    def transcribe_sync(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> RawTranscription:
        total_duration = float(request.options.get("fake_media_duration", 32.0))
        delay = float(request.options.get("fake_delay", 0.0))
        language = request.language or "en"

        progress("loading_model", "Fake model (no download)")
        _sleep_interruptible(0.01, cancel)
        progress("transcribing", None)

        if request.options.get("fake_fail"):
            raise TranscriptionError(
                "Fake provider failure requested for testing.",
                user_message="Fake transcription failure (test).",
            )

        uninterruptible_ms = float(request.options.get("fake_uninterruptible_ms", 0.0))
        if uninterruptible_ms > 0:
            # Simulates a blocking MLX call that cannot observe cancellation.
            time.sleep(uninterruptible_ms / 1000.0)

        segments: list[RawSegment] = []
        segment_span = 4.0
        word_count = 8
        cursor = 0.0
        segment_id = 0
        while cursor < total_duration:
            text = _SCRIPT[segment_id % len(_SCRIPT)]
            tokens = text.split()
            tokens = tokens[:word_count]
            end = min(cursor + segment_span, total_duration)
            span = end - cursor
            words: list[RawWord] = []
            step = span / max(len(tokens), 1)
            for i, token in enumerate(tokens):
                start = cursor + i * step
                words.append(
                    RawWord(text=token, start=round(start, 3), end=round(start + step * 0.85, 3))
                )
            segments.append(
                RawSegment(
                    id=segment_id,
                    start=cursor,
                    end=end,
                    text=text,
                    words=words,
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                )
            )
            segment_id += 1
            cursor = end
            _sleep_interruptible(delay, cancel)

        text = " ".join(s.text for s in segments)
        return RawTranscription(text=text, language=language, segments=segments)


def _sleep_interruptible(seconds: float, cancel: CancellationToken) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        cancel.raise_if_cancelled()
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
