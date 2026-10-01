"""Cloud anchor alignment via OpenRouter.

OpenRouter has no dedicated forced-alignment endpoint, so timing the transcript
of a model that returns plain text or segments is done with a *cloud anchor
pass*: a second, word-timestamp-capable OpenRouter model (MAI-Transcribe 2 by
default, $0.10/hour) transcribes the same audio purely for timing anchors; the
chosen model's transcript tokens are then matched onto those anchor words.
Nothing runs on the local machine, and the transcript text stays 100% from the
model the user selected.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.alignment.base import AlignmentProvider, AlignmentResult
from app.alignment.tiny_anchor import map_remote_tokens
from app.models.result import AlignedWord
from app.providers.base import (
    CancellationToken,
    ProgressCallback,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
)
from app.providers.openrouter import OpenRouterProvider
from app.services.openrouter_models import CLOUD_ANCHOR_MODEL, get_openrouter_model

log = logging.getLogger(__name__)


def _noop(stage: str, message: str | None = None, fraction: float | None = None) -> None:
    pass


class CloudAnchorAligner(AlignmentProvider):
    name = "cloud_anchor"

    def __init__(
        self,
        provider: TranscriptionProvider | None = None,
        anchor_model: str = CLOUD_ANCHOR_MODEL,
    ):
        self._provider = provider
        self._anchor_model = anchor_model
        self._label = get_openrouter_model(anchor_model).display_name

    def align(
        self,
        transcription: RawTranscription,
        media_duration: float | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> AlignmentResult:
        notify = progress or _noop
        warnings: list[str] = []

        audio_path = transcription.meta.get("audio_path")
        if not audio_path or not Path(str(audio_path)).is_file():
            return AlignmentResult(
                words=[],
                warnings=["Cloud alignment needs the prepared audio, which is unavailable."],
            )
        remote_text = transcription.text or " ".join(s.text for s in transcription.segments)
        if not remote_text.strip():
            return AlignmentResult(words=[], warnings=["The transcript is empty."])

        notify("aligning", f"Cloud anchor pass with {self._label}")
        provider = self._provider or OpenRouterProvider()
        request = TranscriptionRequest(
            audio_path=Path(str(audio_path)),
            model_repo=self._anchor_model,
            language=transcription.meta.get("requested_language") or transcription.language,
            context_terms=list(transcription.meta.get("context_terms") or []),
            media_duration=media_duration,
            options={"reuse_existing_chunks": True},
        )

        def anchor_progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
            if stage == "transcribing":
                notify("aligning", f"Cloud anchor pass ({self._label})", fraction)
            elif stage == "loading_model":
                notify("aligning", f"Contacting {self._label}")

        anchor: RawTranscription = provider.transcribe_sync(
            request, anchor_progress, cancel or CancellationToken()
        )

        anchor_words: list[AlignedWord] = []
        for segment in anchor.segments:
            for word in segment.words:
                anchor_words.append(
                    AlignedWord(
                        text=word.text,
                        start=word.start,
                        end=word.end,
                        probability=word.probability,
                        segment_id=segment.id,
                    )
                )
        if not anchor_words:
            warnings.append(
                f"{self._label} returned no word timestamps; subtitle timing was kept as-is."
            )
            return AlignmentResult(words=[], warnings=warnings, meta=self._cost_meta(anchor))

        mapping = map_remote_tokens(remote_text, anchor_words, media_duration)
        warnings.extend(mapping.warnings)
        if not mapping.ok:
            return AlignmentResult(words=[], warnings=warnings, meta=self._cost_meta(anchor))

        log.info(
            "Cloud anchor alignment: %d/%d tokens matched (%.0f%%)",
            int(mapping.matched_ratio * len(mapping.words) or 0),
            len(mapping.words),
            mapping.matched_ratio * 100,
        )
        meta = self._cost_meta(anchor)
        meta["anchor_matched_ratio"] = round(mapping.matched_ratio, 3)
        return AlignmentResult(words=mapping.words, warnings=warnings, meta=meta)

    def _cost_meta(self, anchor: RawTranscription) -> dict:
        return {
            "anchor_model": self._anchor_model,
            "anchor_cost_usd": float(anchor.meta.get("cost_usd", 0.0) or 0.0),
        }
