from __future__ import annotations

import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from app.alignment.base import AlignmentProvider  # noqa: F401  (import parity)
from app.alignment.whisperx_aligner import WhisperXAligner
from app.core.errors import CancelledError
from app.models.result import RawSegment
from app.providers.base import CancellationToken, RawTranscription

STUB_OK = textwrap.dedent(
    """
    import json, sys
    request = json.load(open(sys.argv[1]))
    print("WHISPERX_STAGE: Loading alignment model", flush=True)
    print("WHISPERX_DEVICE: cpu", flush=True)
    print("WHISPERX_STAGE: Aligning words", flush=True)
    words = []
    for i, segment in enumerate(request["segments"]):
        tokens = segment["text"].split()
        if not tokens:
            continue
        span = (segment["end"] - segment["start"]) / len(tokens)
        for j, token in enumerate(tokens):
            words.append({
                "text": token,
                "start": round(segment["start"] + j * span, 3),
                "end": round(segment["start"] + (j + 1) * span, 3),
                "score": 0.9,
            })
        print(f"WHISPERX_PROGRESS: {(i + 1) / len(request['segments']):.4f}", flush=True)
    json.dump({"words": words, "device": "cpu", "language": request["language"]}, open(sys.argv[2], "w"))
    """
)

STUB_FAIL = textwrap.dedent(
    """
    import sys
    print("WHISPERX_ERROR: RuntimeError: model download failed", file=sys.stderr, flush=True)
    sys.exit(1)
    """
)

STUB_SLOW = textwrap.dedent(
    """
    import sys, time
    print("WHISPERX_STAGE: Loading alignment model", flush=True)
    time.sleep(30)
    """
)


def transcription() -> RawTranscription:
    return RawTranscription(
        text="Hello world. Second sentence.",
        language="en",
        segments=[
            RawSegment(id=0, start=0.0, end=2.0, text="Hello world."),
            RawSegment(id=1, start=2.5, end=5.0, text="Second sentence."),
        ],
        meta={"audio_path": "/tmp/whatever.wav"},
    )


def write_stub(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "stub.py"
    path.write_text(source)
    return path


def test_aligns_words_and_reports_progress(tmp_path: Path):
    aligner = WhisperXAligner(
        interpreter=Path(sys.executable), script_path=write_stub(tmp_path, STUB_OK)
    )
    events: list[tuple[str, str | None, float | None]] = []
    result = aligner.align(
        transcription(),
        media_duration=5.0,
        progress=lambda stage, message=None, fraction=None: events.append((stage, message, fraction)),
    )
    assert not result.warnings
    assert [word.text for word in result.words] == ["Hello", "world.", "Second", "sentence."]
    assert result.words[0].start == 0.0
    assert result.words[-1].end == pytest.approx(5.0)
    assert result.words[0].probability == pytest.approx(0.9)
    fractions = [fraction for _, _, fraction in events if fraction is not None]
    assert fractions and max(fractions) == pytest.approx(1.0)


def test_failure_returns_warning_not_exception(tmp_path: Path):
    aligner = WhisperXAligner(
        interpreter=Path(sys.executable), script_path=write_stub(tmp_path, STUB_FAIL)
    )
    result = aligner.align(transcription())
    assert result.words == []
    assert any("failed" in warning for warning in result.warnings)


def test_missing_interpreter_is_reported(tmp_path: Path):
    aligner = WhisperXAligner(
        interpreter=tmp_path / "no-such-python",
        script_path=write_stub(tmp_path, STUB_OK),
    )
    result = aligner.align(transcription())
    assert result.words == []
    assert any("not installed" in warning for warning in result.warnings)


def test_no_segments_is_reported(tmp_path: Path):
    aligner = WhisperXAligner(
        interpreter=Path(sys.executable), script_path=write_stub(tmp_path, STUB_OK)
    )
    empty = RawTranscription(text="hello", segments=[])
    result = aligner.align(empty)
    assert result.words == []
    assert any("segment timing" in warning for warning in result.warnings)


def test_cancellation_terminates_the_subprocess(tmp_path: Path):
    aligner = WhisperXAligner(
        interpreter=Path(sys.executable), script_path=write_stub(tmp_path, STUB_SLOW)
    )
    token = CancellationToken()
    threading.Thread(target=lambda: (time.sleep(0.6), token.cancel()), daemon=True).start()
    with pytest.raises(CancelledError):
        aligner.align(transcription(), cancel=token)


def test_status_reports_missing_environment(monkeypatch, tmp_path: Path):
    from app.services import whisperx as whisperx_service

    monkeypatch.setattr(whisperx_service, "WHISPERX_VENV", tmp_path / "missing-venv")
    whisperx_service.invalidate_status_cache()
    status = whisperx_service.whisperx_status(force=True)
    assert status["installed"] is False
    assert status["venv_exists"] is False


def test_status_detects_installed_probe(monkeypatch, tmp_path: Path):
    from app.services import whisperx as whisperx_service

    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    # Use a wrapper that makes the probe succeed without a real whisperx install.
    probe_ok = tmp_path / "probe_ok.py"
    probe_ok.write_text("print('{\"version\": \"test\", \"device\": \"mps\"}')")
    python = fake_venv / "bin" / "python"
    python.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{probe_ok}'\n")
    python.chmod(0o755)
    monkeypatch.setattr(whisperx_service, "WHISPERX_VENV", fake_venv)
    whisperx_service.invalidate_status_cache()
    status = whisperx_service.whisperx_status(force=True)
    assert status["installed"] is True
    assert status["device"] == "mps"
    whisperx_service.invalidate_status_cache()
