"""Job pipeline: media -> audio -> transcription -> alignment -> subtitles -> files.

Runs inside the worker process (or in-process for tests with injected provider
and aligner). Every stage reports its own status and timing; cancellation is
checked between stages and inside audio extraction.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.alignment.base import AlignmentProvider
from app.alignment.mlx_word_aligner import MLXWordTimestampAligner
from app.alignment.tiny_anchor import build_alignment_windows, map_remote_tokens
from app.alignment.whisperx_aligner import WhisperXAligner
from app.config import AppPaths
from app.exporters import json_exporter, srt, txt, vtt
from app.models.domain import AppSettings, Job, ProviderName, utcnow
from app.models.result import (
    AlignedWord,
    ContextInfo,
    OutputPaths,
    RawSegment,
    StageTimings,
    TranscriptResult,
)
from app.providers import create_provider
from app.providers.base import (
    CancellationToken,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
    noop_progress,
)
from app.providers.local_mlx import LocalMLXProvider
from app.providers.openrouter import segments_from_words
from app.services import media as media_service
from app.services.glossary import build_initial_prompt_from_terms, clean_glossary_terms
from app.services.outputs import create_output_files, release_unwritten, write_text_atomic
from app.services.registry import get_model_spec
from app.services.whisperx import whisperx_status
from app.subtitles.segmenter import (
    SegmenterOptions,
    segment_from_segments,
    segment_words,
)
from app.subtitles.validate import QuantizedCue, quantize_cues, validate_and_repair_cues

log = logging.getLogger(__name__)

StageProgress = Callable[[str, str | None, float | None], None]


def noop_stage_progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
    pass


ANCHOR_MODEL_REPO = "mlx-community/whisper-tiny"


@dataclass
class PipelineOutcome:
    result: TranscriptResult
    outputs: dict[str, str]
    timings: StageTimings
    warnings: list[str] = field(default_factory=list)
    detected_language: str | None = None
    media_duration: float | None = None
    quantized_cues: list[QuantizedCue] = field(default_factory=list)
    provider_meta: dict = field(default_factory=dict)
    alignment_provider: str | None = None


@dataclass
class PipelineContext:
    job: Job
    settings: AppSettings
    paths: AppPaths
    output_dir: Path
    work_dir: Path
    progress: StageProgress = noop_stage_progress
    cancel: CancellationToken = field(default_factory=CancellationToken)
    provider: TranscriptionProvider | None = None
    aligner: AlignmentProvider | None = None
    whisperx_aligner: AlignmentProvider | None = None
    anchor_provider: TranscriptionProvider | None = None
    provider_options: dict = field(default_factory=dict)


def segmenter_options(settings: AppSettings) -> SegmenterOptions:
    prefs = settings.subtitles
    return SegmenterOptions(
        max_line_chars=prefs.max_line_chars,
        max_lines=prefs.max_lines,
        max_cue_duration=prefs.max_cue_duration,
    )


def run_pipeline(ctx: PipelineContext) -> PipelineOutcome:
    job = ctx.job
    cancel = ctx.cancel
    started = time.monotonic()
    timings = StageTimings()
    warnings: list[str] = []
    source_path = Path(job.source_path)

    # --- probe -----------------------------------------------------------
    ctx.progress("preparing", "Reading media")
    cancel.raise_if_cancelled()
    t0 = time.monotonic()
    media_info = media_service.probe_media(source_path)
    media_service.require_audio(media_info)
    timings.media_probe = time.monotonic() - t0
    media_duration = media_info.duration_seconds

    # --- audio preparation ------------------------------------------------
    ctx.progress("extracting_audio", "Preparing 16 kHz audio")
    cancel.raise_if_cancelled()
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = ctx.work_dir / "audio.wav"
    t0 = time.monotonic()

    def on_extract_progress(fraction: float | None) -> None:
        ctx.progress("extracting_audio", "Preparing 16 kHz audio", fraction)

    media_service.extract_audio(
        source_path,
        audio_path,
        media_duration=media_duration,
        cancel=cancel,
        on_progress=on_extract_progress,
    )
    timings.audio_prepare = time.monotonic() - t0
    cancel.raise_if_cancelled()

    # --- transcription -----------------------------------------------------
    provider = ctx.provider or create_provider(job.config.provider.value)
    is_remote = job.config.provider == ProviderName.OPENROUTER
    if is_remote:
        from app.services.openrouter_models import DEFAULT_OPENROUTER_MODEL

        model_repo = job.config.openrouter_model or DEFAULT_OPENROUTER_MODEL
        result_model_key = "openrouter"
    else:
        model_spec = get_model_spec(job.config.model_key)
        model_repo = model_spec.repo_id
        result_model_key = model_spec.key
    context_terms = clean_glossary_terms(job.config.glossary)
    initial_prompt = build_initial_prompt_from_terms(context_terms)

    options = {**ctx.provider_options, **job.config.options}
    if media_duration:
        options.setdefault("fake_media_duration", media_duration)
    request = TranscriptionRequest(
        audio_path=audio_path,
        model_repo=model_repo,
        language=job.config.language.whisper_code,
        initial_prompt=initial_prompt,
        context_terms=context_terms,
        media_duration=media_duration,
        options=options,
    )

    stage_marks: dict[str, float] = {}

    def provider_progress(
        stage: str, message: str | None = None, fraction: float | None = None
    ) -> None:
        now = time.monotonic()
        if stage == "loading_model":
            stage_marks["model_start"] = now
        elif stage == "transcribing" and "model_start" in stage_marks:
            timings.model_load = now - stage_marks.pop("model_start")
        ctx.progress(stage, message, fraction)

    transcription: RawTranscription = provider.transcribe_sync(
        request, provider_progress, cancel
    )
    if "model_start" in stage_marks:  # provider never reported "transcribing"
        timings.model_load = time.monotonic() - stage_marks.pop("model_start")
    timings.inference = transcription.duration or 0.0
    provider_meta = dict(transcription.meta)
    if provider_meta.get("warnings"):
        warnings.extend(provider_meta.pop("warnings"))
    cancel.raise_if_cancelled()

    # --- alignment ---------------------------------------------------------
    ctx.progress("aligning", "Aligning word timestamps")
    t0 = time.monotonic()
    transcription.meta = {**provider_meta, "audio_path": str(audio_path)}
    words, alignment_provider_used, alignment_warnings = _align_transcription(
        ctx, transcription, media_duration, cancel
    )
    timings.alignment = time.monotonic() - t0
    warnings.extend(alignment_warnings)
    cancel.raise_if_cancelled()

    # --- subtitle segmentation --------------------------------------------
    ctx.progress("formatting", "Building subtitles")
    t0 = time.monotonic()
    opts = segmenter_options(ctx.settings)
    if words:
        segmentation = segment_words(words, opts, media_duration)
    else:
        segmentation = segment_from_segments(
            transcription.segments, opts, media_duration
        )
    warnings.extend(segmentation.warnings)
    validation = validate_and_repair_cues(
        segmentation.cues,
        words,
        media_duration,
        max_line_chars=opts.max_line_chars,
        max_lines=opts.max_lines,
    )
    warnings.extend(validation.warnings)
    quantized, quant_warnings = quantize_cues(validation.cues)
    warnings.extend(quant_warnings)
    if not quantized:
        warnings.append("No subtitle cues could be produced for this media.")
    timings.formatting = time.monotonic() - t0
    cancel.raise_if_cancelled()

    # --- saving -------------------------------------------------------------
    ctx.progress("saving", "Saving results")
    t0 = time.monotonic()
    output_dir = ctx.output_dir
    stem, output_paths = create_output_files(output_dir, job.source_filename)
    written: set[str] = set()

    export_segments = transcription.segments
    if not export_segments and words:
        export_segments = segments_from_words(words)

    try:
        write_text_atomic(output_paths["txt"], txt.render(export_segments))
        written.add("txt")
        if quantized:
            write_text_atomic(output_paths["srt"], srt.render(quantized))
            written.add("srt")
            write_text_atomic(output_paths["vtt"], vtt.render(quantized))
            written.add("vtt")

        processing_duration = time.monotonic() - started
        timings.total = round(processing_duration, 3)
        result = TranscriptResult(
            app_version=_app_version(),
            job_id=job.id,
            source_filename=job.source_filename,
            source_path=str(source_path),
            source_metadata=media_info,
            media_duration=media_duration,
            provider=provider.name,
            model_key=result_model_key,
            model_id=model_repo,
            language_requested=job.config.language.value,
            language_detected=transcription.language,
            glossary=job.config.glossary,
            initial_prompt=initial_prompt,
            segments=transcription.segments,
            words=words,
            cues=validation.cues,
            warnings=warnings,
            transcription_started_at=job.started_at or utcnow(),
            processing_duration_seconds=round(processing_duration, 3),
            realtime_factor=(
                round(media_duration / processing_duration, 3)
                if media_duration and processing_duration > 0
                else None
            ),
            timings=timings,
            provider_meta=provider_meta,
            alignment_provider=alignment_provider_used,
            context=ContextInfo(
                global_context=job.config.global_context,
                project_id=job.config.project_id,
                project_name=job.config.project_name,
                per_file_context=job.config.per_file_context,
                effective=job.config.glossary,
                terms=context_terms,
            ),
        )
        result.output_paths = OutputPaths(
            txt=str(output_paths["txt"]),
            srt=str(output_paths["srt"]) if quantized else None,
            vtt=str(output_paths["vtt"]) if quantized else None,
            json_path=str(output_paths["json"]),
        )
        write_text_atomic(output_paths["json"], json_exporter.render(result))
        written.add("json")
        # Remove placeholders for outputs that were not produced (e.g. no cues).
        release_unwritten(output_paths, written)
    except Exception:
        release_unwritten(output_paths, written)
        raise
    timings.saving = time.monotonic() - t0

    outputs = {
        "txt": str(output_paths["txt"]),
        "json": str(output_paths["json"]),
        "basename": stem,
    }
    if quantized:
        outputs["srt"] = str(output_paths["srt"])
        outputs["vtt"] = str(output_paths["vtt"])
    log.info(
        "Job %s finished: %d cues, %.1fs media in %.1fs (%.2fx)",
        job.id,
        len(quantized),
        media_duration or 0,
        processing_duration,
        (media_duration / processing_duration) if media_duration else 0,
    )
    return PipelineOutcome(
        result=result,
        outputs=outputs,
        timings=timings,
        warnings=warnings,
        detected_language=transcription.language,
        media_duration=media_duration,
        quantized_cues=quantized,
        provider_meta=provider_meta,
        alignment_provider=alignment_provider_used,
    )


def _align_transcription(
    ctx: PipelineContext,
    transcription: RawTranscription,
    media_duration: float | None,
    cancel: CancellationToken,
) -> tuple[list[AlignedWord], str | None, list[str]]:
    """Build word timings for this transcription.

    Returns (words, alignment_provider_name, warnings).

    1. Native word timestamps are extracted first (Whisper cross-attention DTW
       or the remote model's own words).
    2. With 'Improve subtitle alignment' enabled, WhisperX re-times the
       transcript against the audio (never re-transcribing it).
    3. When a remote model returned no timing at all, a local Whisper tiny
       anchor pass supplies real speech windows for WhisperX.
    """
    warnings: list[str] = []
    native_aligner = ctx.aligner or MLXWordTimestampAligner()
    native = native_aligner.align(transcription, media_duration, noop_progress, cancel)
    warnings.extend(native.warnings)
    native_words = native.words
    native_provider = "native_word_timestamps" if native_words else None

    if not ctx.job.config.align_with_whisperx:
        return native_words, native_provider, warnings

    status = whisperx_status()
    whisperx = ctx.whisperx_aligner or WhisperXAligner()
    if not (status.get("installed") and whisperx.is_available()):
        warnings.append(
            "WhisperX alignment is not installed; native timestamps were kept. "
            "Install it from Settings."
        )
        return native_words, native_provider, warnings

    def whisperx_progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
        ctx.progress("aligning", message, fraction)

    # Case 1: the provider returned segments - align them directly.
    if transcription.segments:
        aligned = whisperx.align(transcription, media_duration, whisperx_progress, cancel)
        warnings.extend(aligned.warnings)
        if aligned.words:
            return aligned.words, "whisperx", warnings
        return native_words, native_provider, warnings

    # Case 2: words without segments - derive segment windows from the words.
    if native_words:
        synthetic = transcription.model_copy(
            update={"segments": segments_from_words(native_words)}
        )
        aligned = whisperx.align(synthetic, media_duration, whisperx_progress, cancel)
        warnings.extend(aligned.warnings)
        if aligned.words:
            return aligned.words, "whisperx", warnings
        return native_words, native_provider, warnings

    # Case 3: text only - local tiny anchor pass, then WhisperX in those windows.
    ctx.progress("aligning", "Running local speech anchor pass (Whisper tiny)")
    anchor_provider = ctx.anchor_provider or LocalMLXProvider()
    anchor_request = TranscriptionRequest(
        audio_path=Path(str(transcription.meta.get("audio_path", ""))),
        model_repo=ANCHOR_MODEL_REPO,
        language=ctx.job.config.language.whisper_code,
        media_duration=media_duration,
    )

    def anchor_progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
        if stage == "loading_model":
            ctx.progress("aligning", "Loading anchor model (Whisper tiny)", None)
        else:
            ctx.progress("aligning", "Speech anchor pass (Whisper tiny)", fraction)

    anchor_result = anchor_provider.transcribe_sync(anchor_request, anchor_progress, cancel)
    anchor_alignment = MLXWordTimestampAligner().align(
        anchor_result, media_duration, noop_progress, cancel
    )
    if not anchor_alignment.words:
        warnings.append(
            "The local anchor pass produced no word timings; subtitles were not generated."
        )
        return [], None, warnings

    remote_text = transcription.text or " ".join(s.text for s in transcription.segments)
    mapping = map_remote_tokens(remote_text, anchor_alignment.words, media_duration)
    warnings.extend(mapping.warnings)
    if not mapping.ok:
        return [], None, warnings

    windows = build_alignment_windows(mapping.words, anchor_result.segments)
    if windows:
        window_transcription = RawTranscription(
            text=remote_text,
            language=transcription.language,
            segments=[
                RawSegment(id=index, start=w["start"], end=w["end"], text=w["text"])
                for index, w in enumerate(windows)
            ],
            meta=transcription.meta,
        )
        aligned = whisperx.align(window_transcription, media_duration, whisperx_progress, cancel)
        warnings.extend(aligned.warnings)
        if aligned.words:
            return aligned.words, "tiny_anchor+whisperx", warnings
    return mapping.words, "tiny_anchor", warnings


def cleanup_work_dir(work_dir: Path, keep: bool = False) -> None:
    if keep:
        return
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except OSError:
        log.exception("Failed to clean work dir %s", work_dir)


def _app_version() -> str:
    from app.version import APP_VERSION

    return APP_VERSION
