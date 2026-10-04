from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import (
    CancelledError,
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterNoKeyError,
)
from app.models.result import RawWord
from app.providers.base import CancellationToken, TranscriptionRequest
from app.providers.openrouter import OpenRouterProvider, segments_from_words
from tests.conftest import requires_ffmpeg  # noqa: F401


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr("app.providers.openrouter.secrets.load_api_key", lambda: None)


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr("app.providers.openrouter.secrets.load_api_key", lambda: "sk-or-test")


def request_for(audio: Path, **options) -> TranscriptionRequest:
    return TranscriptionRequest(
        audio_path=audio,
        model_repo=options.pop("model", "openai/whisper-large-v3"),
        language=options.pop("language", "en"),
        media_duration=options.pop("duration", 6.0),
        options=options,
    )


def words_payload(offset: float) -> dict:
    return {
        "text": "Welcome to the lecture.",
        "language": "en",
        "duration": 6.0,
        "segments": [
            {"start": 0.0 + offset, "end": 3.0 + offset, "text": "Welcome to the"},
            {"start": 3.0 + offset, "end": 6.0 + offset, "text": "lecture."},
        ],
        "words": [
            {"word": "Welcome", "start": 0.1 + offset, "end": 0.6 + offset},
            {"word": "lecture.", "start": 3.2 + offset, "end": 3.9 + offset},
        ],
        "usage": {"cost": 0.001, "seconds": 6.0},
    }


def noop(stage, message=None, fraction=None):
    pass


def test_missing_key_raises(no_key, tmp_path):
    provider = OpenRouterProvider(base_url="http://mock")
    request = request_for(tmp_path / "missing.wav")
    with pytest.raises(OpenRouterNoKeyError):
        provider.transcribe_sync(request, noop, CancellationToken())


@requires_ffmpeg
def test_verbose_json_single_chunk(with_key, sine_wav, monkeypatch):
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.headers["Authorization"].startswith("Bearer sk-or-test")
        return httpx.Response(200, json=words_payload(0.0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    result = provider.transcribe_sync(request_for(sine_wav), noop, CancellationToken())

    assert len(calls) == 1
    assert calls[0]["response_format"] == "verbose_json"
    assert calls[0]["timestamp_granularities"] == ["segment", "word"]
    assert calls[0]["input_audio"]["format"] == "mp3"
    raw = base64.b64decode(calls[0]["input_audio"]["data"])
    assert raw[:2] == b"\xff\xf3" or raw[:2] == b"\xff\xfb" or len(raw) > 1000  # mp3 payload
    assert result.segments and result.segments[0].text == "Welcome to the"
    assert result.segments[0].words or result.segments[1].words
    assert result.meta["native_word_timestamps"] is True
    assert result.meta["cost_usd"] == pytest.approx(0.001, abs=1e-9)
    assert result.meta["chunks"] == 1


@requires_ffmpeg
def test_chunk_offsets_are_applied(with_key, sine_wav, monkeypatch):
    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "TARGET_CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(chunking, "MAX_CHUNK_SECONDS", 4.0)
    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.4)

    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = counter["n"]
        counter["n"] += 1
        # second chunk response timestamps are chunk-relative
        payload = words_payload(0.0 if index == 0 else 0.0)
        payload["segments"] = [
            {"start": 0.0, "end": 2.5, "text": f"chunk {index}"},
        ]
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    result = provider.transcribe_sync(request_for(sine_wav, duration=6.0), noop, CancellationToken())

    assert counter["n"] >= 2
    starts = [segment.start for segment in result.segments]
    assert starts == sorted(starts)
    assert starts[-1] >= 2.5  # offset of the later chunk applied
    assert result.meta["chunks"] >= 2


@requires_ffmpeg
def test_verbose_rejected_falls_back_to_text_only(with_key, sine_wav):
    verbose_attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("response_format") == "verbose_json":
            verbose_attempts["n"] += 1
            return httpx.Response(400, json={"error": {"message": "verbose_json not supported"}})
        return httpx.Response(
            200,
            json={"text": "Welcome to the lecture.", "usage": {"cost": 0.0005, "seconds": 6.0}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    result = provider.transcribe_sync(request_for(sine_wav), noop, CancellationToken())

    assert verbose_attempts["n"] == 1  # learned after the first chunk
    assert result.segments == []
    assert result.meta["native_word_timestamps"] is False
    assert result.text == "Welcome to the lecture."
    assert any("no timestamps" in warning for warning in result.meta["warnings"])


@requires_ffmpeg
def test_auth_error_is_explicit(with_key, sine_wav):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid key"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    with pytest.raises(OpenRouterAuthError):
        provider.transcribe_sync(request_for(sine_wav), noop, CancellationToken())


@requires_ffmpeg
def test_retries_on_rate_limit(with_key, sine_wav, monkeypatch):
    monkeypatch.setattr("app.providers.openrouter._RETRY_BACKOFF", (0.01, 0.01))
    statuses = [429, 429, 429, 429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(200, json=words_payload(0.0))
        return httpx.Response(status, json={"error": {"message": "rate limited"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    result = provider.transcribe_sync(request_for(sine_wav), noop, CancellationToken())
    assert result.meta["chunks"] == 1
    assert statuses == []


def test_retry_delay_honors_server_hints():
    from app.providers.openrouter import _retry_delay

    plain = httpx.Response(429, json={"error": {"message": "rate limited"}})
    assert _retry_delay(plain, 0) == 2.0
    assert _retry_delay(None, 3) == 15.0

    header = httpx.Response(429, headers={"Retry-After": "7"}, json={})
    assert _retry_delay(header, 0) == 7.0

    metadata = httpx.Response(429, json={"error": {"metadata": {"retry_after_seconds": 3}}})
    assert _retry_delay(metadata, 1) == 5.0  # backoff already larger
    huge = httpx.Response(429, json={"error": {"metadata": {"retry_after_seconds": 60}}})
    assert _retry_delay(huge, 0) == 20.0  # capped


@requires_ffmpeg
def test_cancellation_between_chunks(with_key, sine_wav, monkeypatch):
    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "TARGET_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(chunking, "MAX_CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.4)

    token = CancellationToken()

    def handler(request: httpx.Request) -> httpx.Response:
        token.cancel()  # cancel while the first chunk is in flight
        return httpx.Response(200, json=words_payload(0.0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    with pytest.raises(CancelledError):
        provider.transcribe_sync(request_for(sine_wav, duration=6.0), noop, token)


@requires_ffmpeg
def test_chunks_run_in_parallel_when_configured(with_key, sine_wav, monkeypatch):
    import threading
    import time as time_module

    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "TARGET_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(chunking, "MAX_CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.4)

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time_module.sleep(0.15)
        with lock:
            state["active"] -= 1
        return httpx.Response(200, json=words_payload(0.0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    result = provider.transcribe_sync(
        request_for(sine_wav, chunk_parallelism=3), noop, CancellationToken()
    )

    assert result.meta["chunks"] == 3
    assert state["peak"] >= 2, f"expected overlapping chunk requests, peak={state['peak']}"
    assert len(result.segments) == 6


@requires_ffmpeg
def test_chunk_parallelism_can_be_disabled(with_key, sine_wav, monkeypatch):
    import threading
    import time as time_module

    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "TARGET_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(chunking, "MAX_CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.4)

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time_module.sleep(0.05)
        with lock:
            state["active"] -= 1
        return httpx.Response(200, json=words_payload(0.0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    result = provider.transcribe_sync(
        request_for(sine_wav, chunk_parallelism=1), noop, CancellationToken()
    )

    assert result.meta["chunks"] == 3
    assert state["peak"] == 1
    assert len(result.segments) == 6


def test_chunk_seconds_option_changes_chunk_count(with_key, sine_wav, monkeypatch):
    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.4)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=words_payload(0.0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)

    default_result = provider.transcribe_sync(
        request_for(sine_wav), noop, CancellationToken()
    )
    assert default_result.meta["chunks"] == 1

    short_result = provider.transcribe_sync(
        request_for(sine_wav, chunk_seconds=2), noop, CancellationToken()
    )
    assert short_result.meta["chunks"] >= 2
    assert calls["n"] == 1 + short_result.meta["chunks"]


def test_segments_from_words_groups_on_pauses_and_sentences():
    words = [
        RawWord(text="Hello", start=0.0, end=0.4),
        RawWord(text="world.", start=0.5, end=0.9),
        RawWord(text="After", start=2.5, end=2.9),
        RawWord(text="a", start=3.0, end=3.1),
        RawWord(text="pause.", start=3.2, end=3.6),
    ]
    segments = segments_from_words(words)
    assert len(segments) == 2
    assert segments[0].text == "Hello world."
    assert segments[1].start == 2.5


@requires_ffmpeg
def test_throttled_chunk_is_retried_after_the_rest(with_key, sine_wav, monkeypatch):
    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "TARGET_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(chunking, "MAX_CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.4)
    monkeypatch.setattr("app.providers.openrouter._RETRY_BACKOFF", (0.01,))
    monkeypatch.setattr("app.providers.openrouter._CHUNK_RETRY_PAUSE", 0.01)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 5:
            # Chunk 1 exhausts all five attempts during the first pass.
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=words_payload(0.0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    result = provider.transcribe_sync(request_for(sine_wav), noop, CancellationToken())

    assert result.meta["chunks"] == 3
    assert calls["n"] == 8  # 5 failed attempts + 2 other chunks + retried chunk 1
    assert len(result.segments) == 6


@requires_ffmpeg
def test_persistent_throttling_fails_after_both_passes(with_key, sine_wav, monkeypatch):
    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "TARGET_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(chunking, "MAX_CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.4)
    monkeypatch.setattr("app.providers.openrouter._RETRY_BACKOFF", (0.01,))
    monkeypatch.setattr("app.providers.openrouter._CHUNK_RETRY_PAUSE", 0.01)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    with pytest.raises(OpenRouterError):
        provider.transcribe_sync(request_for(sine_wav), noop, CancellationToken())
    assert calls["n"] == 30  # 3 chunks x 5 attempts x 2 passes
