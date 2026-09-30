"""Transcription provider abstraction.

Everything above this layer (queue, API, exporters) is provider-agnostic: a
future VastProvider or RemoteWhisperProvider only needs to implement this
interface and return the same internal models.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.errors import CancelledError
from app.models.result import RawSegment


class RawTranscription(BaseModel):
    text: str
    language: str | None = None
    segments: list[RawSegment] = Field(default_factory=list)
    duration: float | None = None


@dataclass
class TranscriptionRequest:
    audio_path: Path
    model_repo: str
    language: str | None  # None means auto-detect
    initial_prompt: str | None = None
    options: dict = field(default_factory=dict)


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("Cancelled by user")


class ProgressCallback:
    """Callable reporting a stage transition.

    ``fraction`` is an optional honest sub-progress value in [0, 1] within the
    stage (e.g. byte progress while downloading a model, processed-audio
    fraction while transcribing). ``None`` means the stage is indeterminate.
    """

    def __call__(
        self, stage: str, message: str | None = None, fraction: float | None = None
    ) -> None:
        raise NotImplementedError


def noop_progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
    pass


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
