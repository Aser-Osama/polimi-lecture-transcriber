"""Application error hierarchy.

`user_message` is what the UI shows; `str(error)` carries technical detail that
belongs in the logs.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-facing failures."""

    code = "app_error"
    user_message = "Something went wrong."

    def __init__(self, detail: str = "", user_message: str | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail or self.user_message
        if user_message is not None:
            self.user_message = user_message


class FFmpegMissingError(AppError):
    code = "ffmpeg_missing"
    user_message = (
        "FFmpeg is not installed. Install it with: brew install ffmpeg — then restart the app."
    )


class MediaError(AppError):
    code = "media_error"
    user_message = "The media file could not be read."


class NoAudioTrackError(MediaError):
    code = "no_audio_track"
    user_message = "This file has no audio track, so it cannot be transcribed."


class UnsupportedFormatError(MediaError):
    code = "unsupported_format"
    user_message = "Unsupported media format."


class InsufficientDiskSpaceError(AppError):
    code = "disk_space"
    user_message = "Not enough free disk space for temporary audio processing."


class TranscriptionError(AppError):
    code = "transcription_error"
    user_message = "Transcription failed."


class ModelLoadError(TranscriptionError):
    code = "model_load_error"
    user_message = "The Whisper model could not be loaded. Check your internet connection for the first download."


class ModelDownloadError(AppError):
    code = "model_download_error"
    user_message = "The model download failed. Check your internet connection and try again."


class OutputError(AppError):
    code = "output_error"
    user_message = "The transcript files could not be written. Check the output folder in Settings."


class CancelledError(AppError):
    code = "cancelled"
    user_message = "Cancelled."


class InvalidSourceError(AppError):
    code = "invalid_source"
    user_message = "The selected file does not exist or is not readable."


class WorkerError(AppError):
    code = "worker_error"
    user_message = "The transcription worker stopped unexpectedly."
