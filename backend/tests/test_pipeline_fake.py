from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from app.config import AppPaths
from app.core.errors import CancelledError
from app.models.domain import (
    AppSettings,
    Job,
    JobConfig,
    LanguageChoice,
    ProviderName,
)
from app.models.result import RawSegment
from app.providers.base import (
    CancellationToken,
    RawTranscription,
    TranscriptionProvider,
)
from app.providers.fake import FakeProvider
from app.services.pipeline import PipelineContext, run_pipeline
from tests.conftest import requires_ffmpeg


def make_context(
    paths: AppPaths,
    sine_wav: Path,
    settings: AppSettings | None = None,
    provider: TranscriptionProvider | None = None,
    cancel: CancellationToken | None = None,
    options: dict | None = None,
) -> PipelineContext:
    job = Job(
        source_path=str(sine_wav),
        source_filename=sine_wav.name,
        config=JobConfig(
            provider=ProviderName.FAKE,
            language=LanguageChoice.ENGLISH,
            glossary="NUMA, TLB",
        ),
    )
    return PipelineContext(
        job=job,
        settings=settings or AppSettings(),
        paths=paths,
        output_dir=paths.default_output_dir,
        work_dir=paths.temp_dir / "jobs" / job.id,
        cancel=cancel or CancellationToken(),
        provider=provider or FakeProvider(),
        provider_options=options or {},
    )


@requires_ffmpeg
def test_pipeline_end_to_end_with_fake_provider(paths: AppPaths, sine_wav: Path):
    ctx = make_context(paths, sine_wav)
    outcome = run_pipeline(ctx)

    assert Path(outcome.outputs["txt"]).is_file()
    assert Path(outcome.outputs["srt"]).is_file()
    assert Path(outcome.outputs["vtt"]).is_file()
    assert Path(outcome.outputs["json"]).is_file()
    assert outcome.quantized_cues, "expected subtitle cues"

    srt_text = Path(outcome.outputs["srt"]).read_text(encoding="utf-8")
    assert "00:00:00,000 -->" in srt_text
    index = 1
    for block in srt_text.strip().split("\n\n"):
        lines = block.splitlines()
        assert lines[0] == str(index)
        assert " --> " in lines[1]
        index += 1

    payload = json.loads(Path(outcome.outputs["json"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["job_id"] == ctx.job.id
    assert payload["source_filename"] == sine_wav.name
    assert payload["provider"] == "fake"
    assert payload["model_id"] == "mlx-community/whisper-large-v3-mlx"
    assert payload["segments"], "raw whisper segments must be preserved"
    assert payload["words"], "word-level timestamps must be preserved"
    assert payload["cues"], "final cues must be preserved"
    assert payload["output_paths"]["json"].endswith(".json")
    assert payload["realtime_factor"] is not None
    assert payload["media_duration"] == pytest.approx(6.0, abs=0.2)
    assert payload["transcription_started_at"]
    for word in payload["words"]:
        assert word["start"] < word["end"]


@requires_ffmpeg
def test_pipeline_duplicate_runs_do_not_overwrite(paths: AppPaths, sine_wav: Path):
    first = run_pipeline(make_context(paths, sine_wav))
    second = run_pipeline(make_context(paths, sine_wav))
    assert Path(first.outputs["txt"]).parent == Path(second.outputs["txt"]).parent
    assert Path(first.outputs["txt"]) != Path(second.outputs["txt"])
    assert Path(first.outputs["txt"]).is_file()
    assert Path(second.outputs["txt"]).is_file()
    assert "(2)" in Path(second.outputs["txt"]).name


class WordlessProvider(TranscriptionProvider):
    name = "fake"

    def transcribe_sync(self, request, progress, cancel):
        progress("loading_model", None)
        progress("transcribing", None)
        segments = [
            RawSegment(id=0, start=0.0, end=4.0, text="A fallback segment without words."),
            RawSegment(id=1, start=4.5, end=8.0, text="Another fallback segment."),
        ]
        return RawTranscription(text="joined", language="en", segments=segments)


@requires_ffmpeg
def test_pipeline_falls_back_to_segments_without_words(paths: AppPaths, sine_wav: Path):
    ctx = make_context(paths, sine_wav, provider=WordlessProvider())
    outcome = run_pipeline(ctx)
    payload = json.loads(Path(outcome.outputs["json"]).read_text(encoding="utf-8"))
    assert payload["words"] == []
    assert payload["cues"]
    assert any("word-level timestamps were unavailable" in w for w in payload["warnings"])
    srt_text = Path(outcome.outputs["srt"]).read_text(encoding="utf-8")
    assert "fallback segment" in srt_text


class SlowProvider(FakeProvider):
    def transcribe_sync(self, request, progress, cancel):
        progress("loading_model", None)
        progress("transcribing", None)
        time.sleep(5)
        return super().transcribe_sync(request, progress, cancel)


@requires_ffmpeg
def test_pipeline_cancellation_cleans_up(paths: AppPaths, sine_wav: Path):
    token = CancellationToken()
    ctx = make_context(paths, sine_wav, provider=SlowProvider(), cancel=token)

    def cancel_soon() -> None:
        time.sleep(0.4)
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    with pytest.raises(CancelledError):
        run_pipeline(ctx)
    assert list(paths.default_output_dir.glob("sample*")) == []


@requires_ffmpeg
def test_pipeline_reports_transcription_fractions(paths: AppPaths, sine_wav: Path):
    events: list[tuple[str, str | None, float | None]] = []
    ctx = make_context(paths, sine_wav)
    ctx.progress = lambda stage, message=None, fraction=None: events.append(
        (stage, message, fraction)
    )
    run_pipeline(ctx)
    fractions = [
        fraction for stage, _, fraction in events if stage == "transcribing" and fraction is not None
    ]
    assert fractions, "expected transcription progress fractions"
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)


@requires_ffmpeg
def test_pipeline_empty_media_directory_created(paths: AppPaths, sine_wav: Path):
    ctx = make_context(paths, sine_wav, options={"fake_media_duration": 4.0})
    outcome = run_pipeline(ctx)
    assert outcome.timings.total > 0
    assert outcome.timings.audio_prepare >= 0
    assert outcome.result.language_detected == "en"
