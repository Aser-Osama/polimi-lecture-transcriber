"""Transcript result schema (versioned, Pydantic).

The JSON output written for every job retains all layers:
raw provider output (segments), normalized words, final subtitle cues and
metadata. Subtitles can be regenerated from this file without retranscribing.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.domain import MediaInfo
from app.version import RESULT_SCHEMA_VERSION


class RawWord(BaseModel):
    text: str
    start: float
    end: float
    probability: float | None = None


class RawSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[RawWord] = Field(default_factory=list)
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


class AlignedWord(BaseModel):
    text: str
    start: float
    end: float
    probability: float | None = None
    segment_id: int | None = None


class Cue(BaseModel):
    index: int
    start: float
    end: float
    lines: list[str]
    text: str
    first_word_index: int
    last_word_index: int  # exclusive


class StageTimings(BaseModel):
    media_probe: float = 0.0
    audio_prepare: float = 0.0
    model_load: float = 0.0
    inference: float = 0.0
    alignment: float = 0.0
    formatting: float = 0.0
    saving: float = 0.0
    total: float = 0.0


class OutputPaths(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    txt: str
    srt: str | None = None
    vtt: str | None = None
    json_path: str = Field(alias="json")


class TranscriptResult(BaseModel):
    schema_version: int = RESULT_SCHEMA_VERSION
    app_version: str
    job_id: str
    source_filename: str
    source_path: str
    source_metadata: MediaInfo | None = None
    media_duration: float | None = None
    provider: str
    model_key: str
    model_id: str
    language_requested: str
    language_detected: str | None = None
    glossary: str = ""
    initial_prompt: str | None = None
    segments: list[RawSegment] = Field(default_factory=list)
    words: list[AlignedWord] = Field(default_factory=list)
    cues: list[Cue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    transcription_started_at: datetime | None = None
    processing_duration_seconds: float | None = None
    realtime_factor: float | None = None
    timings: StageTimings = Field(default_factory=StageTimings)
    output_paths: OutputPaths | None = None
    alignment_provider: str | None = None
    provider_meta: dict = Field(default_factory=dict)
