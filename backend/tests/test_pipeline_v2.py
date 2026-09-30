from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.alignment.base import AlignmentProvider, AlignmentResult
from app.config import AppPaths
from app.models.domain import (
    AppSettings,
    Job,
    JobConfig,
    LanguageChoice,
    ProviderName,
)
from app.models.result import AlignedWord, RawSegment, RawWord
from app.providers.base import (
    CancellationToken,
    ProgressCallback,
    RawTranscription,
    TranscriptionProvider,
)
from app.services import pipeline as pipeline_module
from app.services.pipeline import PipelineContext, run_pipeline
from tests.conftest import requires_ffmpeg


class EchoAligner(AlignmentProvider):
    """Test double that re-times a transcript deterministically."""

    name = "stub_whisperx"

    def align(
        self,
        transcription: RawTranscription,
        media_duration: float | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> AlignmentResult:
        words: list[AlignedWord] = []
        for segment in transcription.segments:
            tokens = segment.text.split()
            if not tokens:
                continue
            step = (segment.end - segment.start) / len(tokens)
            for index, token in enumerate(tokens):
                words.append(
                    AlignedWord(
                        text=token,
                        start=round(segment.start + index * step, 3),
                        end=round(segment.start + (index + 1) * step, 3),
                        probability=0.8,
                    )
                )
        return AlignmentResult(words=words)


class TextOnlyProvider(TranscriptionProvider):
    name = "fake"

    def __init__(self, text: str):
        self._text = text

    def transcribe_sync(self, request, progress, cancel):
        progress("loading_model", None)
        progress("transcribing", None)
        return RawTranscription(text=self._text, language="en", segments=[])


class AnchorProvider(TranscriptionProvider):
    """Miniature 'whisper tiny' stand-in with matching words."""

    name = "fake"

    def transcribe_sync(self, request, progress, cancel):
        progress("loading_model", None)
        progress("transcribing", None, 0.5)
        progress("transcribing", None, 1.0)
        segments = [
            RawSegment(
                id=0,
                start=0.0,
                end=2.0,
                text="Hello world.",
                words=[
                    RawWord(text="Hello", start=0.1, end=0.6),
                    RawWord(text="world.", start=0.7, end=1.2),
                ],
            ),
            RawSegment(
                id=1,
                start=2.5,
                end=5.0,
                text="Second sentence.",
                words=[
                    RawWord(text="Second", start=2.6, end=3.2),
                    RawWord(text="sentence.", start=3.3, end=4.1),
                ],
            ),
        ]
        return RawTranscription(
            text="Hello world. Second sentence.",
            language="en",
            segments=segments,
        )


def make_ctx(
    paths: AppPaths,
    sine_wav: Path,
    provider: TranscriptionProvider,
    align: bool,
    whisperx_aligner: AlignmentProvider | None = None,
    anchor_provider: TranscriptionProvider | None = None,
) -> PipelineContext:
    job = Job(
        source_path=str(sine_wav),
        source_filename=sine_wav.name,
        config=JobConfig(
            provider=ProviderName.FAKE,
            language=LanguageChoice.ENGLISH,
            align_with_whisperx=align,
        ),
    )
    return PipelineContext(
        job=job,
        settings=AppSettings(),
        paths=paths,
        output_dir=paths.default_output_dir,
        work_dir=paths.temp_dir / "jobs" / job.id,
        provider=provider,
        whisperx_aligner=whisperx_aligner,
        anchor_provider=anchor_provider,
    )


@pytest.fixture
def whisperx_installed(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "whisperx_status", lambda: {"installed": True, "device": "mps"}
    )


@pytest.fixture
def whisperx_missing(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "whisperx_status", lambda: {"installed": False, "device": None}
    )


@requires_ffmpeg
def test_context_flows_to_prompt_and_result(paths: AppPaths, sine_wav: Path):
    from app.providers.fake import FakeProvider

    ctx = make_ctx(paths, sine_wav, FakeProvider(), align=False)
    ctx.job.config.glossary = "NUMA, TLB\nDaniele Cattaneo"
    ctx.job.config.global_context = "NUMA"
    ctx.job.config.per_file_context = "Daniele Cattaneo"
    ctx.job.config.project_name = "OS"
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["initial_prompt"] is not None
    assert "NUMA" in payload["initial_prompt"]
    assert payload["context"]["terms"] == ["NUMA", "TLB", "Daniele Cattaneo"]
    assert payload["context"]["project_name"] == "OS"
    assert payload["context"]["global_context"] == "NUMA"
    assert payload["context"]["per_file_context"] == "Daniele Cattaneo"


@requires_ffmpeg
def test_no_context_produces_no_prompt(paths: AppPaths, sine_wav: Path):
    from app.providers.fake import FakeProvider

    ctx = make_ctx(paths, sine_wav, FakeProvider(), align=False)
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["initial_prompt"] is None
    assert payload["context"]["terms"] == []


@requires_ffmpeg
def test_text_only_without_alignment_produces_no_subtitles(paths: AppPaths, sine_wav: Path):
    ctx = make_ctx(paths, sine_wav, TextOnlyProvider("Just some words with no timing."), align=False)
    outcome = run_pipeline(ctx)
    assert Path(outcome.outputs["txt"]).is_file()
    assert Path(outcome.outputs["json"]).is_file()
    assert "srt" not in outcome.outputs and "vtt" not in outcome.outputs
    assert outcome.quantized_cues == []
    assert any("timestamps" in warning for warning in outcome.warnings)


@requires_ffmpeg
def test_whisperx_refines_native_words(
    paths: AppPaths, sine_wav: Path, whisperx_installed
):
    from app.providers.fake import FakeProvider

    ctx = make_ctx(
        paths,
        sine_wav,
        FakeProvider(),
        align=True,
        whisperx_aligner=EchoAligner(),
    )
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["alignment_provider"] == "whisperx"
    assert outcome.quantized_cues
    assert Path(outcome.outputs["srt"]).is_file()


@requires_ffmpeg
def test_whisperx_missing_keeps_native_timestamps(
    paths: AppPaths, sine_wav: Path, whisperx_missing
):
    from app.providers.fake import FakeProvider

    ctx = make_ctx(paths, sine_wav, FakeProvider(), align=True)
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["alignment_provider"] == "native_word_timestamps"
    assert any("not installed" in warning for warning in outcome.warnings)
    assert outcome.quantized_cues


@requires_ffmpeg
def test_text_only_with_alignment_uses_tiny_anchor_then_whisperx(
    paths: AppPaths, sine_wav: Path, whisperx_installed
):
    ctx = make_ctx(
        paths,
        sine_wav,
        TextOnlyProvider("Hello world. Second sentence."),
        align=True,
        whisperx_aligner=EchoAligner(),
        anchor_provider=AnchorProvider(),
    )
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text())
    assert payload["alignment_provider"] == "tiny_anchor+whisperx"
    assert [word["text"] for word in payload["words"]] == [
        "Hello",
        "world.",
        "Second",
        "sentence.",
    ]
    assert Path(outcome.outputs["srt"]).is_file()
    assert outcome.quantized_cues


@requires_ffmpeg
def test_text_only_with_missing_anchor_match_yields_no_subtitles(
    paths: AppPaths, sine_wav: Path, whisperx_installed
):
    ctx = make_ctx(
        paths,
        sine_wav,
        TextOnlyProvider("completely unrelated content that matches nothing"),
        align=True,
        whisperx_aligner=EchoAligner(),
        anchor_provider=AnchorProvider(),
    )
    outcome = run_pipeline(ctx)
    assert outcome.quantized_cues == []
    assert "srt" not in outcome.outputs
    assert any("did not match" in warning for warning in outcome.warnings)
