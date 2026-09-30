"""Alignment abstraction.

Transcription and timestamp alignment are separate concerns: a provider may
return text without reliable word timing (or none at all), and the aligner
turns the provider output into normalized word-level timestamps. Future
implementations could use WhisperX or wav2vec2 forced alignment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.models.result import AlignedWord
from app.providers.base import CancellationToken, ProgressCallback, RawTranscription

NOOP_PROGRESS: ProgressCallback = lambda stage, message=None: None  # noqa: E731


@dataclass
class AlignmentResult:
    words: list[AlignedWord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AlignmentProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def align(
        self,
        transcription: RawTranscription,
        media_duration: float | None = None,
        progress: ProgressCallback = NOOP_PROGRESS,
        cancel: CancellationToken | None = None,
    ) -> AlignmentResult:
        ...
