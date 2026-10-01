from __future__ import annotations

import contextlib
from pathlib import Path

from app.alignment.cloud_anchor import CloudAnchorAligner
from app.models.result import RawSegment, RawWord
from app.providers.base import (
    CancellationToken,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
)


class AnchorStub(TranscriptionProvider):
    name = "openrouter"

    def __init__(self, segments: list[RawSegment], cost: float = 0.001, language: str = "en"):
        self._segments = segments
        self._cost = cost
        self._language = language
        self.requests: list[TranscriptionRequest] = []

    def transcribe_sync(self, request, progress, cancel):
        self.requests.append(request)
        progress("loading_model", None)
        progress("transcribing", None, 0.5)
        progress("transcribing", None, 1.0)
        return RawTranscription(
            text=" ".join(s.text for s in self._segments),
            language=self._language,
            segments=self._segments,
            meta={"cost_usd": self._cost},
        )


def anchor_segments() -> list[RawSegment]:
    return [
        RawSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world. Second sentence.",
            words=[
                RawWord(text="Hello", start=0.1, end=0.5),
                RawWord(text="world.", start=0.6, end=1.1),
                RawWord(text="Second", start=1.3, end=1.7),
                RawWord(text="sentence.", start=1.75, end=2.2),
            ],
        )
    ]


def primary_text_only() -> RawTranscription:
    return RawTranscription(
        text="Hello world. Second sentence.",
        language="en",
        segments=[],
        meta={"audio_path": __file__},  # any existing file satisfies the check
    )


def test_maps_primary_transcript_onto_cloud_anchor_words(tmp_path: Path):
    stub = AnchorStub(anchor_segments())
    aligner = CloudAnchorAligner(provider=stub)
    events: list[tuple[str, str | None, float | None]] = []
    result = aligner.align(
        primary_text_only(),
        media_duration=2.5,
        progress=lambda stage, message=None, fraction=None: events.append((stage, message, fraction)),
    )
    assert [word.text for word in result.words] == ["Hello", "world.", "Second", "sentence."]
    assert result.words[0].start == 0.1
    assert result.meta["anchor_cost_usd"] == 0.001
    assert result.meta["anchor_model"] == "microsoft/mai-transcribe-2"
    assert result.meta["anchor_matched_ratio"] == 1.0
    stages = [stage for stage, _, _ in events]
    assert stages and all(stage == "aligning" for stage in stages)
    assert stub.requests and stub.requests[0].options.get("reuse_existing_chunks") is True


def test_missing_audio_path_warns():
    aligner = CloudAnchorAligner(provider=AnchorStub(anchor_segments()))
    transcription = primary_text_only()
    transcription.meta = {}
    result = aligner.align(transcription)
    assert result.words == []
    assert any("audio" in warning.lower() for warning in result.warnings)


def test_anchor_without_words_degrades():
    segments = [RawSegment(id=0, start=0.0, end=2.0, text="Hello world.", words=[])]
    aligner = CloudAnchorAligner(provider=AnchorStub(segments))
    result = aligner.align(primary_text_only())
    assert result.words == []
    assert any("no word timestamps" in warning for warning in result.warnings)
    assert result.meta["anchor_cost_usd"] == 0.001


def test_low_match_degrades():
    segments = [
        RawSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="completely different content",
            words=[
                RawWord(text="completely", start=0.1, end=0.4),
                RawWord(text="different", start=0.5, end=0.9),
                RawWord(text="content", start=1.0, end=1.4),
            ],
        )
    ]
    aligner = CloudAnchorAligner(provider=AnchorStub(segments))
    result = aligner.align(primary_text_only())
    assert result.words == []
    assert any("did not match" in warning for warning in result.warnings)


def test_cancellation_token_is_forwarded():
    class CancellingStub(AnchorStub):
        def transcribe_sync(self, request, progress, cancel):
            cancel.cancel()
            raise AssertionError("provider should see the cancelled token")

    aligner = CloudAnchorAligner(provider=CancellingStub(anchor_segments()))
    token = CancellationToken()
    token.cancel()
    with contextlib.suppress(AssertionError):
        aligner.align(primary_text_only(), cancel=token)
        # expected: the stub observed cancellation
