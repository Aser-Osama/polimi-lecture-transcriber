"""OpenRouter speech-to-text provider.

Verified against the OpenRouter STT API (September 2026):
- POST {base}/audio/transcriptions with a JSON body carrying base64 audio
  (`input_audio: {data, format}`). Base64 avoids the 25 MB multipart cap.
- `response_format=verbose_json` + `timestamp_granularities=["segment","word"]`
  returns segments and (where supported) words. Providers that cannot do
  structured output reject verbose_json with 400; we then retry with plain
  json and record a warning (text-only result).
- Upstream providers time out after ~60 s per request, so the prepared audio
  is split into silence-aligned chunks of ~5 minutes and stitched back with
  offsets. Chunk progress is genuine (chunk i of n).

Usage cost is returned by OpenRouter and preserved in the result metadata.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

from app.core.errors import (
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterNoKeyError,
)
from app.models.result import AlignedWord, RawSegment, RawWord
from app.providers.base import (
    CancellationToken,
    ProgressCallback,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
)
from app.services import secrets
from app.services.chunking import Chunk, prepare_chunks
from app.services.openrouter_models import get_openrouter_model

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF = (2.0, 5.0)


def _default_base_url() -> str:
    return os.environ.get("PT_OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def segments_from_words(
    words: list[RawWord] | list[AlignedWord], max_words: int = 40, pause: float = 0.9
) -> list[RawSegment]:
    """Group real word timings into segments (split at pauses/sentence ends).

    Used when a provider returns words without segments; the boundaries come
    from the actual word timestamps, nothing is fabricated.
    """
    if not words:
        return []
    sentences = (".", "!", "?", "…", "。", "！", "？")
    segments: list[RawSegment] = []
    current: list[RawWord] = []

    def flush() -> None:
        if not current:
            return
        segments.append(
            RawSegment(
                id=len(segments),
                start=round(current[0].start, 3),
                end=round(current[-1].end, 3),
                text=" ".join(w.text for w in current),
                words=[
                    RawWord(text=w.text, start=w.start, end=w.end, probability=w.probability)
                    for w in current
                ],
            )
        )
        current.clear()

    for word in words:
        if current:
            gap = word.start - current[-1].end
            ends_sentence = current[-1].text.rstrip("\"')]}\u201d\u2019").endswith(sentences)
            if gap >= pause or (ends_sentence and len(current) >= 3) or len(current) >= max_words:
                flush()
        current.append(word)
    flush()
    return segments


def _attach_words_to_segments(segments: list[RawSegment], words: list[RawWord]) -> None:
    """Distribute flat words into their containing segments (by midpoint)."""
    if not segments:
        return
    boundaries = [(segment.start + segment.end) / 2 for segment in segments]
    for word in words:
        midpoint = (word.start + word.end) / 2
        best = 0
        for index, boundary in enumerate(boundaries):
            if midpoint >= boundary:
                best = index
            else:
                break
        # clamp to the nearest segment when outside the covered range
        if midpoint < segments[0].start:
            best = 0
        elif midpoint > segments[-1].end:
            best = len(segments) - 1
        segments[best].words.append(word)


class OpenRouterProvider(TranscriptionProvider):
    name = "openrouter"

    def __init__(self, base_url: str | None = None, http_client=None):
        self._base_url = (base_url or _default_base_url()).rstrip("/")
        self._client = http_client  # injectable for tests
        self._verbose_supported: bool | None = None  # learned at runtime

    # ------------------------------------------------------------- internals

    def _post_chunk(
        self,
        client,
        model_id: str,
        chunk: Chunk,
        language: str | None,
        api_key: str,
        verbose: bool,
        context_terms: list[str] | None = None,
        context_mode: str = "none",
    ) -> tuple[dict, bool]:
        """POST one chunk. Returns (payload, verbose_still_supported)."""
        audio_b64 = base64.b64encode(chunk.path.read_bytes()).decode("ascii")
        body: dict[str, Any] = {
            "model": model_id,
            "input_audio": {"data": audio_b64, "format": "mp3"},
        }
        if language:
            body["language"] = language
        if verbose:
            body["response_format"] = "verbose_json"
            body["timestamp_granularities"] = ["segment", "word"]
        if context_terms and context_mode == "phrase_list":
            # MAI-Transcribe keyword biasing (provider-specific option).
            provider = body.setdefault("provider", {})
            options = provider.setdefault("options", {})
            azure = options.setdefault("azure", {})
            azure["phraseList"] = {"phrases": context_terms[:100]}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Polimi Lecture Transcriber",
        }

        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = client.post(
                    f"{self._base_url}/audio/transcriptions", json=body, headers=headers
                )
            except Exception as exc:  # httpx transport errors
                last_error = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)])
                    continue
                raise OpenRouterError(
                    f"OpenRouter request failed: {exc}",
                    user_message="Could not reach OpenRouter. Check your internet connection.",
                ) from exc

            if response.status_code == 401 or response.status_code == 403:
                raise OpenRouterAuthError(
                    f"OpenRouter auth failed ({response.status_code}): {response.text[:200]}"
                )
            if response.status_code == 402:
                raise OpenRouterError(
                    f"OpenRouter payment required: {response.text[:200]}",
                    user_message="OpenRouter reports insufficient credits for this account.",
                )
            if response.status_code == 400 and verbose:
                # Provider cannot do structured output; fall back to plain JSON.
                return self._post_chunk(
                    client,
                    model_id,
                    chunk,
                    language,
                    api_key,
                    verbose=False,
                    context_terms=context_terms,
                    context_mode=context_mode,
                )
            if response.status_code in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)])
                continue
            if response.status_code != 200:
                raise OpenRouterError(
                    f"OpenRouter error {response.status_code}: {response.text[:300]}",
                    user_message=f"OpenRouter request failed (HTTP {response.status_code}).",
                )
            try:
                return response.json(), verbose
            except ValueError as exc:
                raise OpenRouterError(
                    f"OpenRouter returned invalid JSON: {exc}",
                    user_message="OpenRouter returned an unexpected response.",
                ) from exc
        raise OpenRouterError(
            f"OpenRouter request failed after retries: {last_error}",
            user_message="OpenRouter request failed after several attempts.",
        )

    @staticmethod
    def _merge_payload(
        payload: dict,
        chunk: Chunk,
        segments: list[RawSegment],
        words: list[RawWord],
        texts: list[str],
        languages: list[str],
        usage: dict,
    ) -> None:
        offset = chunk.start
        for raw_segment in payload.get("segments") or []:
            try:
                start = float(raw_segment["start"]) + offset
                end = float(raw_segment["end"]) + offset
            except (KeyError, TypeError, ValueError):
                continue
            segments.append(
                RawSegment(
                    id=len(segments),
                    start=round(start, 3),
                    end=round(end, 3),
                    text=str(raw_segment.get("text", "")).strip(),
                )
            )
        for raw_word in payload.get("words") or []:
            try:
                start = float(raw_word["start"]) + offset
                end = float(raw_word["end"]) + offset
            except (KeyError, TypeError, ValueError):
                continue
            text = str(raw_word.get("word", "")).strip()
            if not text:
                continue
            words.append(RawWord(text=text, start=round(start, 3), end=round(end, 3)))
        text = str(payload.get("text") or "").strip()
        if text:
            texts.append(text)
        language = payload.get("language")
        if isinstance(language, str) and language:
            languages.append(language)
        chunk_usage = payload.get("usage") or {}
        if isinstance(chunk_usage, dict):
            for key in ("cost", "seconds", "total_tokens", "input_tokens", "output_tokens"):
                value = chunk_usage.get(key)
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + float(value)

    # ------------------------------------------------------------ interface

    def transcribe_sync(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> RawTranscription:
        cancel.raise_if_cancelled()
        spec = get_openrouter_model(request.model_repo)
        api_key = secrets.load_api_key()
        if not api_key:
            raise OpenRouterNoKeyError("No OpenRouter API key stored in Keychain")

        import httpx

        work_dir = request.audio_path.parent

        def on_chunk_prep(fraction: float) -> None:
            progress("transcribing", f"Preparing chunks for upload ({fraction * 100:.0f}%)")

        chunks = prepare_chunks(
            request.audio_path,
            work_dir,
            request.media_duration,
            cancel=cancel,
            on_progress=on_chunk_prep,
        )
        cancel.raise_if_cancelled()
        log.info(
            "OpenRouter: %s, %d chunk(s), total %.0fs",
            spec.model_id,
            len(chunks),
            sum(c.duration for c in chunks),
        )

        warnings: list[str] = []
        segments: list[RawSegment] = []
        words: list[RawWord] = []
        texts: list[str] = []
        languages: list[str] = []
        usage: dict[str, float] = {}
        text_only_chunks = 0
        started = time.monotonic()

        context_terms = [term for term in (request.context_terms or []) if term.strip()][:100]
        context_applied = "none"
        if context_terms:
            if spec.context == "phrase_list":
                context_applied = "phrase_list"
            else:
                warnings.append(
                    f"{spec.display_name} does not support context biasing; "
                    f"{len(context_terms)} context term(s) were not sent."
                )

        def run(client) -> None:
            nonlocal text_only_chunks
            for position, chunk in enumerate(chunks, start=1):
                cancel.raise_if_cancelled()
                use_verbose = self._verbose_supported is not False
                progress(
                    "transcribing",
                    f"Transcribing with {spec.display_name} - chunk {position}/{len(chunks)}",
                    fraction=(position - 1) / len(chunks),
                )
                payload, verbose_ok = self._post_chunk(
                    client,
                    spec.model_id,
                    chunk,
                    request.language,
                    api_key,
                    use_verbose,
                    context_terms=context_terms,
                    context_mode=spec.context,
                )
                if use_verbose and not verbose_ok:
                    self._verbose_supported = False
                elif use_verbose and verbose_ok:
                    self._verbose_supported = True
                if not payload.get("segments") and not payload.get("words"):
                    text_only_chunks += 1
                self._merge_payload(payload, chunk, segments, words, texts, languages, usage)
                progress(
                    "transcribing",
                    f"Transcribed chunk {position}/{len(chunks)}",
                    fraction=position / len(chunks),
                )
                cancel.raise_if_cancelled()

        if self._client is not None:
            run(self._client)
        else:
            timeout = httpx.Timeout(connect=20.0, read=180.0, write=120.0, pool=20.0)
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                run(client)

        elapsed = time.monotonic() - started

        if text_only_chunks:
            warnings.append(
                f"{spec.display_name} returned no timestamps for {text_only_chunks} chunk(s). "
                "Enable 'Improve subtitle alignment' (or a different model) for subtitles."
            )
        if words and not segments:
            segments = segments_from_words(words)
        elif words and segments:
            _attach_words_to_segments(segments, words)
        if not words and segments:
            warnings.append(
                f"{spec.display_name} returned segment timestamps without word timestamps; "
                "word timings can be refined with WhisperX alignment."
            )

        segments = [s for s in segments if s.text]

        meta = {
            "openrouter_model": spec.model_id,
            "chunks": len(chunks),
            "native_word_timestamps": bool(words),
            "native_segments": bool(segments),
            "cost_usd": round(usage.get("cost", 0.0), 6),
            "audio_seconds": round(usage.get("seconds", sum(c.duration for c in chunks)), 2),
            "context_applied": context_applied,
            "context_terms": len(context_terms),
        }
        # Keep reasoning about joins consistent: join chunk texts with spaces.
        full_text = " ".join(texts)
        if not full_text:
            full_text = " ".join(s.text for s in segments)

        return RawTranscription(
            text=full_text,
            language=languages[0] if languages else (request.language or None),
            segments=segments,
            duration=elapsed,
            meta={**meta, "warnings": warnings},
        )
