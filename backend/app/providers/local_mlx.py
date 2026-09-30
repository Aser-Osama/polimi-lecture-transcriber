"""Local MLX Whisper provider for Apple Silicon.

Verified against mlx-whisper 0.4.3:
- ``mlx_whisper.transcribe(...)`` supports ``word_timestamps=True`` which
  returns per-word ``{word, start, end, probability}`` via cross-attention +
  DTW alignment.
- ``initial_prompt`` and ``language`` pass through ``**decode_options``.
- ``mlx_whisper.transcribe.ModelHolder`` caches the last loaded model per
  path, so a persistent worker process reuses the model across batch jobs.
- The model is loaded through ``ModelHolder.get_model`` before transcription
  so model-load time can be measured separately from inference time.

Audio is handed to mlx-whisper as an in-memory float32 array (16 kHz mono),
so no uncancellable FFmpeg subprocess runs inside the library.
"""

from __future__ import annotations

import logging
import time

from app.core.errors import CancelledError, ModelLoadError, TranscriptionError
from app.models.result import RawSegment, RawWord
from app.providers.base import (
    CancellationToken,
    ProgressCallback,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
)
from app.providers.tqdm_progress import hook_tqdm
from app.services.audioio import load_wav_float32
from app.services.model_cache import download_model, is_model_cached

log = logging.getLogger(__name__)


class LocalMLXProvider(TranscriptionProvider):
    name = "local_mlx"

    def transcribe_sync(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> RawTranscription:
        cancel.raise_if_cancelled()

        try:
            import mlx.core as mx
            import mlx_whisper
            from mlx_whisper.transcribe import ModelHolder
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TranscriptionError(
                f"mlx-whisper is not importable: {exc}",
                user_message="MLX Whisper is not available in this Python environment. "
                "Run ./setup.sh to reinstall dependencies.",
            ) from exc

        if is_model_cached(request.model_repo):
            progress("loading_model", "Loading model from cache")
        else:
            # Download first so the UI gets real byte-level progress instead of
            # an indeterminate "downloading..." state.
            progress("loading_model", "Downloading model (first use) - this can take a while")
            download_model(
                request.model_repo,
                progress=lambda fraction, message: progress("loading_model", message, fraction),
                cancel=cancel,
            )

        # Load (or reuse from cache) the model explicitly so that model-load
        # time is measurable and separate from inference time.
        try:
            ModelHolder.get_model(request.model_repo, mx.float16)
        except Exception as exc:
            log.exception("Model load failed for %s", request.model_repo)
            raise ModelLoadError(
                f"Failed to load model {request.model_repo}: {exc}",
            ) from exc

        cancel.raise_if_cancelled()
        progress("transcribing", None)

        def report_audio_fraction(fraction: float) -> None:
            cancel.raise_if_cancelled()
            progress("transcribing", None, fraction)

        try:
            audio = load_wav_float32(request.audio_path)
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to read prepared audio {request.audio_path}: {exc}",
                user_message="The prepared audio file could not be read.",
            ) from exc

        started = time.monotonic()
        try:
            # mlx-whisper's own tqdm counts processed mel frames; forward that
            # as genuine progress for this stage.
            with hook_tqdm(report_audio_fraction):
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=request.model_repo,
                    language=request.language,
                    initial_prompt=request.initial_prompt,
                    word_timestamps=True,
                    verbose=None,
                    condition_on_previous_text=True,
                )
        except CancelledError:
            raise
        except Exception as exc:
            log.exception("mlx_whisper.transcribe failed")
            raise TranscriptionError(
                f"mlx_whisper.transcribe failed: {exc}",
                user_message="Whisper transcription failed. See logs for details.",
            ) from exc
        inference_seconds = time.monotonic() - started

        cancel.raise_if_cancelled()

        segments = [
            RawSegment(
                id=int(seg.get("id", index)),
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=str(seg.get("text", "")).strip(),
                temperature=_opt_float(seg.get("temperature")),
                avg_logprob=_opt_float(seg.get("avg_logprob")),
                compression_ratio=_opt_float(seg.get("compression_ratio")),
                no_speech_prob=_opt_float(seg.get("no_speech_prob")),
                words=[
                    RawWord(
                        text=str(w["word"]),
                        start=float(w["start"]),
                        end=float(w["end"]),
                        probability=_opt_float(w.get("probability")),
                    )
                    for w in seg.get("words", [])
                    if "start" in w and "end" in w and str(w.get("word", "")).strip()
                ],
            )
            for index, seg in enumerate(result.get("segments", []))
        ]

        return RawTranscription(
            text=str(result.get("text", "")).strip(),
            language=result.get("language"),
            segments=segments,
            duration=inference_seconds,
            meta={
                "context_applied": "initial_prompt" if request.initial_prompt else "none",
                "context_terms": len(request.context_terms or []),
            },
        )


def _opt_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
