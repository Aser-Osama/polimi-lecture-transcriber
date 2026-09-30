from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from app.core.errors import CancelledError, MediaError, NoAudioTrackError
from app.providers.base import CancellationToken
from app.services import media as media_service
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_probe_audio_file(sine_wav: Path):
    info = media_service.probe_media(sine_wav)
    assert info.has_audio
    assert not info.has_video
    assert info.audio_codec == "pcm_s16le"
    assert info.sample_rate == 16000
    assert info.duration_seconds == pytest.approx(6.0, abs=0.1)
    assert info.size_bytes > 0


@requires_ffmpeg
def test_probe_video_without_audio(silent_video: Path):
    info = media_service.probe_media(silent_video)
    assert info.has_video
    assert not info.has_audio
    with pytest.raises(NoAudioTrackError):
        media_service.require_audio(info)


@requires_ffmpeg
def test_probe_missing_file(tmp_path: Path):
    with pytest.raises(MediaError):
        media_service.probe_media(tmp_path / "nope.mp4")


@requires_ffmpeg
def test_probe_corrupted_file(tmp_path: Path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(os.urandom(4096))
    with pytest.raises(MediaError):
        media_service.probe_media(bad)


@requires_ffmpeg
def test_probe_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(MediaError):
        media_service.probe_media(empty)


@requires_ffmpeg
def test_extract_audio_produces_16k_mono(sine_wav: Path, tmp_path: Path):
    destination = tmp_path / "audio.wav"
    progress: list[float | None] = []
    media_service.extract_audio(
        sine_wav, destination, media_duration=6.0, on_progress=progress.append
    )
    assert destination.exists()
    import wave

    with wave.open(str(destination)) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() / 16000 == pytest.approx(6.0, abs=0.1)
    assert progress, "expected progress callbacks"
    assert progress[-1] == 1.0


@requires_ffmpeg
def test_extract_audio_cancellation(tmp_path: Path):
    fifo = tmp_path / "blocking.fifo"
    os.mkfifo(fifo)
    destination = tmp_path / "audio.wav"
    token = CancellationToken()

    result: dict = {}

    def run() -> None:
        try:
            media_service.extract_audio(fifo, destination, cancel=token)
            result["status"] = "completed"
        except CancelledError:
            result["status"] = "cancelled"
        except Exception as exc:  # pragma: no cover
            result["status"] = "error"
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.5)
    token.cancel()
    thread.join(timeout=15)
    assert not thread.is_alive(), "extraction did not stop after cancellation"
    assert result.get("status") == "cancelled"
    assert not destination.exists()


@requires_ffmpeg
def test_tool_available():
    ok, info = media_service.tool_available("ffmpeg")
    assert ok and info and "ffmpeg" in info.lower()


@requires_ffmpeg
def test_disk_space_check_ok(tmp_path: Path):
    media_service.check_disk_space(tmp_path, media_duration=1.0)


def test_disk_space_check_insufficient(tmp_path: Path, monkeypatch):
    import shutil

    class Usage:
        free = 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda _: Usage())
    from app.core.errors import InsufficientDiskSpaceError

    with pytest.raises(InsufficientDiskSpaceError):
        media_service.check_disk_space(tmp_path, media_duration=3600.0)
