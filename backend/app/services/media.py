"""Media probing (ffprobe) and audio preparation (ffmpeg).

All subprocesses use argument arrays. Audio extraction is cancellable: the
FFmpeg child is polled and terminated on request, and it runs in the worker's
process group so a forced worker kill takes it down as well.
"""

from __future__ import annotations

import json
import logging
import os
import queue as std_queue
import select
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from app import capabilities
from app.core.errors import (
    FFmpegMissingError,
    InsufficientDiskSpaceError,
    MediaError,
    NoAudioTrackError,
    UnsupportedFormatError,
)
from app.core.progress import CancellationToken
from app.models.domain import MediaInfo, MediaStreamInfo

log = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
_TARGET_SAMPLE_RATE = 16000
_BYTES_PER_SECOND_PCM16 = _TARGET_SAMPLE_RATE * 2


def tool_available(name: str) -> tuple[bool, str | None]:
    """Return (available, version-string-or-error)."""
    try:
        result = subprocess.run(
            [name, "-version"], capture_output=True, text=True, timeout=15, check=False
        )
    except FileNotFoundError:
        return False, f"{name} not found in PATH"
    except OSError as exc:
        return False, f"{name} could not be executed: {exc}"
    if result.returncode != 0:
        return False, f"{name} exited with {result.returncode}"
    first_line = (result.stdout or result.stderr).splitlines()
    return True, first_line[0] if first_line else "unknown version"


def check_ffmpeg() -> tuple[bool, str | None]:
    ffmpeg_ok, ffmpeg_info = tool_available(FFMPEG)
    if not ffmpeg_ok:
        return False, ffmpeg_info
    probe_ok, probe_info = tool_available(FFPROBE)
    if not probe_ok:
        return False, probe_info
    return True, ffmpeg_info


def ffmpeg_install_hint() -> str:
    platform = capabilities.sys_platform()
    if platform == "darwin":
        return "brew install ffmpeg"
    if platform == "win32":
        return "winget install Gyan.FFmpeg"
    return "sudo apt install ffmpeg (or your distribution's package manager)"


def ensure_ffmpeg() -> None:
    ok, info = check_ffmpeg()
    if not ok:
        raise FFmpegMissingError(
            f"FFmpeg preflight failed: {info}",
            user_message=f"FFmpeg is not installed. Install it with: {ffmpeg_install_hint()} — "
            "then restart the app.",
        )


def probe_media(path: Path) -> MediaInfo:
    ensure_ffmpeg()
    if not path.exists():
        raise MediaError(f"File does not exist: {path}", user_message="The file does not exist.")
    if not path.is_file():
        raise MediaError(f"Not a file: {path}", user_message="The selected path is not a file.")
    if not path.stat().st_size:
        raise MediaError(f"Empty file: {path}", user_message="The selected file is empty.")

    try:
        result = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegMissingError(str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            f"ffprobe timed out on {path}", user_message="Reading the media file timed out."
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or "").strip()[-500:]
        raise MediaError(
            f"ffprobe failed for {path}: {detail}",
            user_message="This file could not be read as media. It may be corrupted or unsupported.",
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc

    streams: list[MediaStreamInfo] = []
    for stream in payload.get("streams", []):
        codec_type = stream.get("codec_type", "unknown")
        disposition = stream.get("disposition") or {}
        streams.append(
            MediaStreamInfo(
                index=int(stream.get("index", -1)),
                codec_type=codec_type,
                codec_name=stream.get("codec_name"),
                sample_rate=_safe_int(stream.get("sample_rate")),
                channels=_safe_int(stream.get("channels")),
                language=(stream.get("tags") or {}).get("language"),
                width=_safe_int(stream.get("width")),
                height=_safe_int(stream.get("height")),
                is_attached_picture=bool(disposition.get("attached_pic")),
            )
        )

    audio_streams = [s for s in streams if s.codec_type == "audio"]
    video_streams = [s for s in streams if s.codec_type == "video" and not s.is_attached_picture]
    if not audio_streams and not video_streams:
        raise UnsupportedFormatError(
            f"No audio or video streams in {path}", user_message="No audio or video found in this file."
        )

    fmt = payload.get("format") or {}
    duration = _safe_float(fmt.get("duration"))
    if duration is None and audio_streams:
        duration = _safe_float((payload.get("streams") or [{}])[audio_streams[0].index].get("duration"))

    primary_audio = audio_streams[0] if audio_streams else None
    primary_video = video_streams[0] if video_streams else None

    return MediaInfo(
        path=str(path),
        filename=path.name,
        size_bytes=path.stat().st_size,
        duration_seconds=duration,
        format_name=fmt.get("format_name"),
        has_audio=bool(audio_streams),
        has_video=bool(video_streams),
        audio_codec=primary_audio.codec_name if primary_audio else None,
        video_codec=primary_video.codec_name if primary_video else None,
        sample_rate=primary_audio.sample_rate if primary_audio else None,
        channels=primary_audio.channels if primary_audio else None,
        bit_rate=_safe_int(fmt.get("bit_rate")),
        streams=streams,
    )


def require_audio(media: MediaInfo) -> None:
    if not media.has_audio:
        raise NoAudioTrackError(f"{media.path} has no audio stream")


def check_disk_space(target_dir: Path, media_duration: float | None) -> None:
    expected = 64 * 1024 * 1024  # baseline for short clips
    if media_duration:
        expected += int(media_duration * _BYTES_PER_SECOND_PCM16 * 1.3)
    free = shutil.disk_usage(target_dir).free
    if free < expected:
        raise InsufficientDiskSpaceError(
            f"Need ~{expected} bytes free in {target_dir}, only {free} available",
        )


def extract_audio(
    source: Path,
    destination: Path,
    media_duration: float | None = None,
    cancel: CancellationToken | None = None,
    on_progress: Callable[[float | None], None] | None = None,
) -> None:
    """Extract 16 kHz mono PCM16 WAV suitable for Whisper."""
    ensure_ffmpeg()
    token = cancel or CancellationToken()
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_disk_space(destination.parent, media_duration)

    args = [
        FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(_TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        "-f",
        "wav",
        str(destination),
    ]
    log.info("Extracting audio: %s", destination.name)
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stderr_chunks: list[str] = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_chunks.append(process.stderr.read() if process.stderr else ""),
        daemon=True,
    )
    stderr_thread.start()

    try:
        for raw_line in _progress_lines(process, token):
            line = raw_line.strip()
            if line.startswith("out_time_us=") and on_progress:
                us = line.split("=", 1)[1]
                try:
                    seconds = int(us) / 1_000_000
                except ValueError:
                    continue
                fraction = None
                if media_duration and media_duration > 0:
                    fraction = max(0.0, min(1.0, seconds / media_duration))
                on_progress(fraction)
            elif line == "progress=end" and on_progress:
                on_progress(1.0)
        if token.cancelled:
            raise _cancelled()
        returncode = process.wait(timeout=30)
    except BaseException:
        _terminate(process)
        raise
    finally:
        stderr_thread.join(timeout=5)
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    if returncode != 0:
        detail = "".join(stderr_chunks).strip()[-800:]
        destination.unlink(missing_ok=True)
        raise MediaError(
            f"ffmpeg audio extraction failed ({returncode}): {detail}",
            user_message="Audio could not be extracted from this file. It may be corrupted.",
        )
    if not destination.exists() or destination.stat().st_size == 0:
        raise MediaError(
            "ffmpeg reported success but produced no audio",
            user_message="No audio could be extracted from this file.",
        )


def _progress_lines(process: subprocess.Popen, token: CancellationToken) -> Iterator[str]:
    """Yield ffmpeg ``-progress`` lines, checking cancellation between reads.

    POSIX: poll the pipe with ``select``. Windows: ``select`` cannot poll pipes,
    so a reader thread feeds a queue while the loop polls for cancellation.
    """
    assert process.stdout is not None
    if os.name == "nt":
        lines: std_queue.Queue[str | None] = std_queue.Queue()

        def reader() -> None:
            try:
                for line in process.stdout:  # type: ignore[union-attr]
                    lines.put(line)
            except ValueError:
                pass
            finally:
                lines.put(None)

        threading.Thread(target=reader, daemon=True).start()
        while True:
            if token.cancelled:
                return
            try:
                line = lines.get(timeout=0.2)
            except std_queue.Empty:
                if process.poll() is not None:
                    return
                continue
            if line is None:
                return
            yield line
    else:
        while True:
            if token.cancelled:
                return
            ready, _, _ = select.select([process.stdout], [], [], 0.2)
            if not ready:
                if process.poll() is not None:
                    return
                continue
            line = process.stdout.readline()
            if not line:
                return
            yield line


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _cancelled():
    from app.core.errors import CancelledError

    return CancelledError("Audio extraction cancelled")


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result
