"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.domain import LanguageChoice, MediaInfo, ProviderName
from app.services.registry import DEFAULT_MODEL_KEY


class UploadResponse(BaseModel):
    upload_id: str
    path: str
    filename: str
    size_bytes: int
    media: MediaInfo


class PickResult(BaseModel):
    path: str
    filename: str
    size_bytes: int
    media: MediaInfo


class PickResponse(BaseModel):
    files: list[PickResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    cancelled: bool = False


class ProbeRequest(BaseModel):
    path: str


class JobSource(BaseModel):
    path: str | None = None
    upload_id: str | None = None


class CreateJobsRequest(BaseModel):
    sources: list[JobSource] = Field(min_length=1, max_length=200)
    model_key: str = DEFAULT_MODEL_KEY
    language: LanguageChoice = LanguageChoice.ENGLISH
    glossary: str | None = None  # legacy alias for per_file_context
    provider: ProviderName = ProviderName.LOCAL_MLX
    openrouter_model: str | None = None
    align_with_whisperx: bool = False
    project_id: str | None = None
    per_file_context: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    context: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    context: str | None = None


class OpenRouterKeyRequest(BaseModel):
    key: str


class AlignmentInstallState(BaseModel):
    state: str = "idle"  # idle|installing|completed|failed
    message: str | None = None


class RegenerateRequest(BaseModel):
    job_id: str | None = None
    json_path: str | None = None
    overwrite: bool = False


class RevealRequest(BaseModel):
    target: str


class ModelDownloadState(BaseModel):
    key: str
    state: str = "idle"  # idle|downloading|completed|failed
    fraction: float | None = None
    message: str | None = None
    error: str | None = None
