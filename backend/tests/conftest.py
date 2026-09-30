from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import AppPaths
from app.db.database import Database

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed"
)


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    paths = AppPaths(
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        temp_dir=tmp_path / "temp",
        default_output_dir=tmp_path / "out",
        db_path=tmp_path / "data" / "transcriber.db",
    )
    paths.ensure_directories()
    return paths


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture(scope="session")
def sine_wav(tmp_path_factory) -> Path:
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")
    directory = tmp_path_factory.mktemp("media")
    path = directory / "sample.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory) -> Path:
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")
    directory = tmp_path_factory.mktemp("media")
    path = directory / "no_audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )
    return path
