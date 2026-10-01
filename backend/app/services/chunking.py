"""Split prepared audio into compressed chunks for remote transcription.

OpenRouter documents a 60-second upstream provider timeout per request, so
long lectures must be split. Cuts are placed at detected silences near the
target chunk length so words are not sliced in half. Chunks are encoded to
small mono MP3 (64 kbps) for base64 upload.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.errors import MediaError
from app.services.media import ensure_ffmpeg

if TYPE_CHECKING:
    from app.providers.base import CancellationToken

log = logging.getLogger(__name__)

TARGET_CHUNK_SECONDS = 300.0  # ~5 min keeps every request well under the timeout
MAX_CHUNK_SECONDS = 360.0
CUT_SEARCH_WINDOW = 45.0
SILENCE_NOISE_DB = "-35dB"
SILENCE_MIN_DURATION = 0.35
MIN_USEFUL_CHUNK = 30.0

_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    path: Path

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_silences(
    audio_path: Path, noise_db: str = SILENCE_NOISE_DB, min_duration: float = SILENCE_MIN_DURATION
) -> list[tuple[float, float]]:
    ensure_ffmpeg()
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={noise_db}:d={min_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    silences: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in (result.stderr or "").splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END.search(line)
        if end_match and pending_start is not None:
            silences.append((pending_start, float(end_match.group(1))))
            pending_start = None
    if pending_start is not None:
        silences.append((pending_start, float("inf")))  # silence runs to EOF
    return silences


def plan_bounds(
    duration: float,
    silences: list[tuple[float, float]],
    target: float = TARGET_CHUNK_SECONDS,
    max_len: float = MAX_CHUNK_SECONDS,
    window: float = CUT_SEARCH_WINDOW,
) -> list[tuple[float, float]]:
    if duration <= max_len:
        return [(0.0, duration)]

    min_useful = min(MIN_USEFUL_CHUNK, max_len / 2.0)
    cuts: list[float] = []
    position = target
    while position < duration - min_useful:
        cut = _choose_cut(position, silences, window)
        if cuts and cut - cuts[-1] > max_len:
            cut = cuts[-1] + max_len
        if cut >= duration - min_useful:
            break
        cuts.append(cut)
        position = cut + target

    boundaries = [0.0, *cuts, duration]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _choose_cut(position: float, silences: list[tuple[float, float]], window: float) -> float:
    best: float | None = None
    best_distance = float("inf")
    for start, end in silences:
        midpoint = start + min(end - start, 2.0) / 2 if end != float("inf") else start + 1.0
        if abs(midpoint - position) <= window and abs(midpoint - position) < best_distance:
            best = midpoint
            best_distance = abs(midpoint - position)
    return best if best is not None else position


def encode_chunk(
    audio_path: Path, destination: Path, start: float, end: float, bitrate: str = "64k"
) -> None:
    ensure_ffmpeg()
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            bitrate,
            "-f",
            "mp3",
            "-y",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        raise MediaError(
            f"ffmpeg chunk encode failed ({result.returncode}): {(result.stderr or '').strip()[-400:]}",
            user_message="Audio could not be prepared for upload.",
        )


def prepare_chunks(
    audio_path: Path,
    work_dir: Path,
    duration: float | None,
    cancel: CancellationToken | None = None,
    on_progress: Callable[[float], None] | None = None,
    reuse_existing: bool = False,
) -> list[Chunk]:
    """Detect silences, plan boundaries and encode every chunk to MP3.

    ``reuse_existing`` skips re-encoding when the chunk file already exists,
    which makes a second pass (e.g. cloud anchor alignment) nearly free.
    """
    from app.config import debug_mode

    if duration is None or duration <= 0:
        raise MediaError(
            "Cannot chunk audio without a known duration",
            user_message="The media duration is unknown, so it cannot be uploaded.",
        )
    silences = detect_silences(audio_path)
    if cancel is not None:
        cancel.raise_if_cancelled()
    bounds = plan_bounds(
        duration,
        silences,
        target=TARGET_CHUNK_SECONDS,
        max_len=MAX_CHUNK_SECONDS,
        window=CUT_SEARCH_WINDOW,
    )
    log.info("Planned %d chunk(s) for %.0fs audio", len(bounds), duration)

    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(bounds, start=1):
        if cancel is not None:
            cancel.raise_if_cancelled()
        path = chunks_dir / f"chunk_{index:03d}.mp3"
        if not ((debug_mode() or reuse_existing) and path.exists()):
            encode_chunk(audio_path, path, start, end)
        chunks.append(Chunk(index=index, start=start, end=end, path=path))
        if on_progress is not None:
            on_progress(index / len(bounds))
    return chunks
