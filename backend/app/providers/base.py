"""Transcription provider abstraction.

Everything above this layer (queue, API, exporters) is provider-agnostic: a
future VastProvider or RemoteWhisperProvider only needs to implement this
interface and return the same internal models.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.progress import (  # noqa: F401 - re-exported for existing imports
    CancellationToken,
    ProgressCallback,
    noop_progress,
)
from app.models.result import RawSegment

__all__ = [
    "CancellationToken",
    "ProgressCallback",
    "RawTranscription",
    "TranscriptionProvider",
    "TranscriptionRequest",
    "noop_progress",
]


class RawTranscription(BaseModel):
    text: str
    language: str | None = None
    segments: list[RawSegment] = Field(default_factory=list)
    duration: float | None = None
    meta: dict = Field(default_factory=dict)


@dataclass
class TranscriptionRequest:
    audio_path: Path
    model_repo: str
    language: str | None  # None means auto-detect
    initial_prompt: str | None = None
    context_terms: list[str] = field(default_factory=list)
    media_duration: float | None = None
    options: dict = field(default_factory=dict)


class TranscriptionProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def transcribe_sync(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> RawTranscription:
        """Blocking transcription. Runs inside the worker process."""

    async def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback = noop_progress,
        cancel: CancellationToken | None = None,
    ) -> RawTranscription:
        token = cancel or CancellationToken()
        return await asyncio.to_thread(self.transcribe_sync, request, progress, token)
