"""Alignment modes: cloud (OpenRouter), local WhisperX, none."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.alignment.base import AlignmentProvider, AlignmentResult
from app.config import AppPaths
from app.models.domain import AppSettings, Job, JobConfig, LanguageChoice, ProviderName
from app.models.result import AlignedWord, RawSegment, RawWord
from app.providers.base import (
    CancellationToken,
    RawTranscription,
    TranscriptionProvider,
)
from app.services.pipeline import PipelineContext, run_pipeline
from tests.conftest import requires_ffmpeg


class WordlessCloudProvider(TranscriptionProvider):
    """Simulates an OpenRouter model without timestamps (metadata included)."""

    name = "openrouter"

    def transcribe_sync(self, request, progress, cancel):
        progress("loading_model", None)
        progress("transcribing", None)
        return RawTranscription(
            text="Hello world. Second sentence.",
            language="en",
            segments=[],
            meta={"openrouter_model": "qwen/qwen3-asr-flash-2026-02-10"},
        )


class NativeCloudProvider(TranscriptionProvider):
    """Simulates MAI-Transcribe 2: OpenRouter with native word timestamps."""

    name = "openrouter"

    def transcribe_sync(self, request, progress, cancel):
        progress("loading_model", None)
        progress("transcribing", None)
        return RawTranscription(
            text="Hello world.",
            language="en",
            segments=[
                RawSegment(
                    id=0,
                    start=0.0,
                    end=1.2,
                    text="Hello world.",
                    words=[
                        RawWord(text="Hello", start=0.1, end=0.5),
                        RawWord(text="world.", start=0.6, end=1.1),
                    ],
                )
            ],
            meta={"openrouter_model": "microsoft/mai-transcribe-2", "cost_usd": 0.0004},
        )


class CountingCloudAligner(AlignmentProvider):
    name = "cloud_anchor"

    def __init__(self):
        self.calls = 0

    def align(self, transcription, media_duration=None, progress=None, cancel=None):
        self.calls += 1
        return AlignmentResult(
            words=[
                AlignedWord(text="Hello", start=0.15, end=0.55),
                AlignedWord(text="world.", start=0.6, end=1.0),
                AlignedWord(text="Second", start=1.2, end=1.6),
                AlignedWord(text="sentence.", start=1.65, end=2.1),
            ],
            meta={"anchor_model": "stub", "anchor_cost_usd": 0.0009},
        )


def make_ctx(paths: AppPaths, sine_wav: Path, provider: TranscriptionProvider, mode: str) -> PipelineContext:
    job = Job(
        source_path=str(sine_wav),
        source_filename=sine_wav.name,
        config=JobConfig(
            provider=ProviderName.OPENROUTER,
            language=LanguageChoice.ENGLISH,
            alignment_mode=mode,
        ),
    )
    return PipelineContext(
        job=job,
        settings=AppSettings(),
        paths=paths,
        output_dir=paths.default_output_dir,
        work_dir=paths.temp_dir / "jobs" / job.id,
        provider=provider,
    )


@requires_ffmpeg
def test_cloud_mode_prefers_native_words_without_extra_pass(paths: AppPaths, sine_wav: Path):
    ctx = make_ctx(paths, sine_wav, NativeCloudProvider(), "cloud")
    cloud_aligner = CountingCloudAligner()
    ctx.cloud_aligner = cloud_aligner
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["alignment_provider"] == "native_word_timestamps"
    assert cloud_aligner.calls == 0, "native timestamps must not trigger a cloud pass"
    assert payload["provider_meta"]["cost_usd"] == pytest.approx(0.0004)
    assert outcome.quantized_cues


@requires_ffmpeg
def test_cloud_mode_uses_anchors_when_model_has_no_timestamps(paths: AppPaths, sine_wav: Path):
    ctx = make_ctx(paths, sine_wav, WordlessCloudProvider(), "cloud")
    cloud_aligner = CountingCloudAligner()
    ctx.cloud_aligner = cloud_aligner
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["alignment_provider"] == "cloud_anchor"
    assert cloud_aligner.calls == 1
    assert payload["provider_meta"]["cost_usd"] == pytest.approx(0.0009)
    assert Path(outcome.outputs["srt"]).is_file()
    assert [word["text"] for word in payload["words"]] == [
        "Hello",
        "world.",
        "Second",
        "sentence.",
    ]


@requires_ffmpeg
def test_cloud_mode_falls_back_for_local_provider(paths: AppPaths, sine_wav: Path):
    from app.providers.fake import FakeProvider

    ctx = make_ctx(paths, sine_wav, FakeProvider(), "cloud")
    # local backend really means no openrouter_model metadata
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["alignment_provider"] == "native_word_timestamps"
    assert any("OpenRouter backend" in warning for warning in outcome.warnings)


@requires_ffmpeg
def test_none_mode_ignores_alignment_helpers(paths: AppPaths, sine_wav: Path, monkeypatch):
    monkeypatch.setattr(
        "app.services.pipeline.whisperx_status",
        lambda: {"installed": True, "device": "mps"},
    )
    ctx = make_ctx(paths, sine_wav, WordlessCloudProvider(), "none")
    ctx.cloud_aligner = CountingCloudAligner()
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["alignment_provider"] is None
    assert ctx.cloud_aligner.calls == 0
    assert "srt" not in outcome.outputs


@requires_ffmpeg
def test_cancellation_propagates_progress_through_alignment(paths: AppPaths, sine_wav: Path):
    ctx = make_ctx(paths, sine_wav, WordlessCloudProvider(), "cloud")
    events: list[tuple[str, str | None, float | None]] = []

    def progress(stage: str, message: str | None = None, fraction: float | None = None):
        events.append((stage, message, fraction))

    ctx.progress = progress
    ctx.cancel = CancellationToken()
    ctx.cloud_aligner = CountingCloudAligner()
    run_pipeline(ctx)
    assert any(stage == "aligning" for stage, _, _ in events)
