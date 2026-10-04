from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.services.chunking import (
    detect_silences,
    encode_chunk,
    plan_bounds,
    prepare_chunks,
)
from tests.conftest import requires_ffmpeg


def test_plan_bounds_no_silences():
    bounds = plan_bounds(700.0, [])
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == pytest.approx(700.0)
    for start, end in bounds:
        assert end - start <= 360.0 + 1e-9
    for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:], strict=False):
        assert prev_end == next_start


def test_plan_bounds_short_audio_single_chunk():
    assert plan_bounds(120.0, []) == [(0.0, 120.0)]


def test_plan_bounds_prefers_silence_near_target():
    silences = [(295.0, 299.0), (320.0, 322.0)]
    bounds = plan_bounds(600.0, silences, target=300.0, max_len=360.0, window=45.0)
    first_end = bounds[0][1]
    assert 295.0 <= first_end <= 322.0


def test_plan_bounds_ignores_silence_outside_window():
    silences = [(100.0, 102.0)]
    bounds = plan_bounds(600.0, silences, target=300.0, max_len=360.0, window=30.0)
    assert bounds[0][1] == pytest.approx(300.0)


def test_plan_bounds_enforces_max_length_without_silence():
    bounds = plan_bounds(1000.0, [], target=300.0, max_len=360.0, window=45.0)
    for start, end in bounds:
        assert end - start <= 360.0 + 1e-9


@requires_ffmpeg
def test_detect_silences_finds_the_gap(tmp_path: Path):
    audio = tmp_path / "gaps.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(audio),
        ],
        check=True,
    )
    silences = detect_silences(audio)
    assert any(1.6 <= start <= 2.4 and 3.6 <= end <= 4.4 for start, end in silences)


@requires_ffmpeg
def test_encode_chunk_produces_mp3(tmp_path: Path, sine_wav: Path):
    destination = tmp_path / "chunk.mp3"
    encode_chunk(sine_wav, destination, 1.0, 3.0)
    assert destination.exists() and destination.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(destination)],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(2.0, abs=0.15)


@requires_ffmpeg
def test_prepare_chunks_splits_long_audio(tmp_path: Path, sine_wav: Path, monkeypatch):
    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "TARGET_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(chunking, "MAX_CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.5)
    work = tmp_path / "work"
    chunks = prepare_chunks(sine_wav, work, duration=6.0)
    assert len(chunks) >= 2
    assert chunks[0].start == 0.0
    assert chunks[-1].end == pytest.approx(6.0)
    for chunk in chunks:
        assert chunk.path.exists()
        assert chunk.duration <= 3.0 + 1e-9


def test_prepare_chunks_target_seconds_override(tmp_path: Path, sine_wav: Path, monkeypatch):
    import app.services.chunking as chunking

    monkeypatch.setattr(chunking, "CUT_SEARCH_WINDOW", 0.5)
    work = tmp_path / "work"
    chunks = prepare_chunks(sine_wav, work, duration=6.0, target_seconds=2.0)
    assert len(chunks) >= 2
    assert chunks[-1].end == pytest.approx(6.0)
    for chunk in chunks:
        assert chunk.duration <= 2.4 + 1e-9  # max is 1.2x the target
