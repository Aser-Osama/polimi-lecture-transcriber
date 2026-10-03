"""Core domain models: jobs, media metadata, settings."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]


class JobStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    EXTRACTING_AUDIO = "extracting_audio"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    FORMATTING = "formatting"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


ACTIVE_STATUSES = frozenset(
    {
        JobStatus.PREPARING,
        JobStatus.EXTRACTING_AUDIO,
        JobStatus.LOADING_MODEL,
        JobStatus.TRANSCRIBING,
        JobStatus.ALIGNING,
        JobStatus.FORMATTING,
        JobStatus.SAVING,
    }
)
TERMINAL_STATUSES = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.INTERRUPTED,
    }
)

STATUS_LABELS: dict[JobStatus, str] = {
    JobStatus.QUEUED: "Queued",
    JobStatus.PREPARING: "Reading media",
    JobStatus.EXTRACTING_AUDIO: "Preparing audio",
    JobStatus.LOADING_MODEL: "Loading model",
    JobStatus.TRANSCRIBING: "Transcribing",
    JobStatus.ALIGNING: "Aligning words",
    JobStatus.FORMATTING: "Formatting subtitles",
    JobStatus.SAVING: "Saving results",
    JobStatus.COMPLETED: "Completed",
    JobStatus.FAILED: "Failed",
    JobStatus.CANCELLED: "Cancelled",
    JobStatus.INTERRUPTED: "Interrupted",
}


class LanguageChoice(str, Enum):
    ENGLISH = "en"
    ITALIAN = "it"
    AUTO = "auto"

    @property
    def whisper_code(self) -> str | None:
        return None if self is LanguageChoice.AUTO else self.value


class ProviderName(str, Enum):
    LOCAL_MLX = "local_mlx"
    OPENROUTER = "openrouter"
    FAKE = "fake"

    @property
    def display_name(self) -> str:
        return {
            ProviderName.LOCAL_MLX: "Local",
            ProviderName.OPENROUTER: "OpenRouter",
            ProviderName.FAKE: "Fake (development)",
        }[self]


class MediaStreamInfo(BaseModel):
    index: int
    codec_type: str
    codec_name: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    language: str | None = None
    width: int | None = None
    height: int | None = None
    is_attached_picture: bool = False


class MediaInfo(BaseModel):
    path: str
    filename: str
    size_bytes: int
    duration_seconds: float | None = None
    format_name: str | None = None
    has_audio: bool = False
    has_video: bool = False
    audio_codec: str | None = None
    video_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None
    streams: list[MediaStreamInfo] = Field(default_factory=list)


class JobConfig(BaseModel):
    model_key: str = "quality"
    language: LanguageChoice = LanguageChoice.ENGLISH
    provider: ProviderName = ProviderName.LOCAL_MLX
    openrouter_model: str | None = None
    align_with_whisperx: bool = False  # legacy flag; see alignment_mode
    alignment_mode: str = "none"  # none | local_whisperx | cloud
    # glossary holds the *effective* merged context (global + project + file).
    glossary: str = ""
    global_context: str = ""
    per_file_context: str = ""
    project_id: str | None = None
    project_name: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_key")
    @classmethod
    def _known_model(cls, value: str) -> str:
        from app.services.registry import MODEL_CATALOG

        if value not in MODEL_CATALOG:
            raise ValueError(f"unknown model key: {value}")
        return value

    @field_validator("openrouter_model")
    @classmethod
    def _known_openrouter_model(cls, value: str | None) -> str | None:
        from app.services.openrouter_models import is_known_openrouter_model

        if value is not None and not is_known_openrouter_model(value):
            raise ValueError(f"unknown OpenRouter model: {value}")
        return value

    @field_validator("alignment_mode")
    @classmethod
    def _known_alignment_mode(cls, value: str) -> str:
        if value not in {"none", "local_whisperx", "cloud"}:
            raise ValueError(f"unknown alignment mode: {value}")
        return value

    @model_validator(mode="after")
    def _openrouter_default(self) -> JobConfig:
        from app.services.openrouter_models import DEFAULT_OPENROUTER_MODEL

        if self.provider == ProviderName.OPENROUTER and not self.openrouter_model:
            self.openrouter_model = DEFAULT_OPENROUTER_MODEL
        return self

    @property
    def effective_alignment_mode(self) -> str:
        """Alignment mode with the legacy boolean flag folded in."""
        if self.alignment_mode in {"local_whisperx", "cloud"}:
            return self.alignment_mode
        return "local_whisperx" if self.align_with_whisperx else "none"


class Project(BaseModel):
    """A course/project holding recurring context (terms, names, acronyms)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    context: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Job(BaseModel):
    id: str = Field(default_factory=new_job_id)
    source_path: str
    source_filename: str
    source_is_temporary: bool = False
    size_bytes: int | None = None
    media: MediaInfo | None = None
    config: JobConfig = Field(default_factory=JobConfig)
    status: JobStatus = JobStatus.QUEUED
    status_message: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_duration: float | None = None
    realtime_factor: float | None = None
    detected_language: str | None = None
    error: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    archived: bool = False

    def public_view(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SubtitlePreferences(BaseModel):
    max_line_chars: int = Field(default=42, ge=20, le=80)
    max_lines: int = Field(default=2, ge=1, le=4)
    max_cue_duration: float = Field(default=7.0, ge=2.0, le=20.0)


class AppSettings(BaseModel):
    output_dir: str | None = None
    default_model_key: str = "quality"
    default_language: LanguageChoice = LanguageChoice.ENGLISH
    default_provider: ProviderName = ProviderName.LOCAL_MLX
    default_openrouter_model: str = "microsoft/mai-transcribe-2"
    default_align_with_whisperx: bool = False
    default_alignment_mode: str = "none"  # none | local_whisperx | cloud
    max_parallel_cloud_jobs: int = Field(default=6, ge=1, le=32)
    glossary: str = ""
    subtitles: SubtitlePreferences = Field(default_factory=SubtitlePreferences)
    keep_temp_uploads: bool = False
    reveal_outputs_on_finish: bool = False

    @field_validator("default_model_key")
    @classmethod
    def _known_model(cls, value: str) -> str:
        from app.services.registry import MODEL_CATALOG

        if value not in MODEL_CATALOG:
            raise ValueError(f"unknown model key: {value}")
        return value

    @field_validator("default_openrouter_model")
    @classmethod
    def _known_openrouter_model(cls, value: str) -> str:
        from app.services.openrouter_models import is_known_openrouter_model

        if not is_known_openrouter_model(value):
            raise ValueError(f"unknown OpenRouter model: {value}")
        return value

    @field_validator("default_alignment_mode")
    @classmethod
    def _known_default_alignment(cls, value: str) -> str:
        if value not in {"none", "local_whisperx", "cloud"}:
            raise ValueError(f"unknown alignment mode: {value}")
        return value
