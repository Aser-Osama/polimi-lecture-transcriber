"""OpenRouter speech-to-text model catalog.

Slugs verified against openrouter.ai (September 2026). All of them are served
through the same endpoint: POST {base}/audio/transcriptions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenRouterModelSpec:
    model_id: str
    display_name: str
    description: str
    timestamps: str  # "words" | "segments" | "none" (best-known capability)


OPENROUTER_MODELS: tuple[OpenRouterModelSpec, ...] = (
    OpenRouterModelSpec(
        model_id="microsoft/mai-transcribe-2",
        display_name="MAI-Transcribe 2 (recommended)",
        description="Microsoft multilingual STT, word timestamps, strong on long-form audio.",
        timestamps="words",
    ),
    OpenRouterModelSpec(
        model_id="openai/whisper-large-v3",
        display_name="Whisper Large V3",
        description="OpenAI Whisper large-v3 through OpenRouter providers.",
        timestamps="words",
    ),
    OpenRouterModelSpec(
        model_id="openai/whisper-large-v3-turbo",
        display_name="Whisper Large V3 Turbo",
        description="Faster Whisper large-v3 Turbo variant.",
        timestamps="words",
    ),
    OpenRouterModelSpec(
        model_id="qwen/qwen3-asr-1.7b",
        display_name="Qwen3 ASR 1.7B",
        description="Qwen multilingual ASR with segment and word timestamps.",
        timestamps="words",
    ),
    OpenRouterModelSpec(
        model_id="qwen/qwen3-asr-flash-2026-02-10",
        display_name="Qwen3 ASR Flash",
        description="Fast Qwen ASR; may return plain text only (alignment can add timing).",
        timestamps="unknown",
    ),
)

DEFAULT_OPENROUTER_MODEL = "microsoft/mai-transcribe-2"

_BY_ID = {spec.model_id: spec for spec in OPENROUTER_MODELS}


def is_known_openrouter_model(model_id: str) -> bool:
    return model_id in _BY_ID


def get_openrouter_model(model_id: str) -> OpenRouterModelSpec:
    try:
        return _BY_ID[model_id]
    except KeyError:
        raise KeyError(f"Unknown OpenRouter model: {model_id}") from None


def openrouter_model_choices() -> list[dict]:
    return [
        {
            "id": spec.model_id,
            "display_name": spec.display_name,
            "description": spec.description,
            "timestamps": spec.timestamps,
            "default": spec.model_id == DEFAULT_OPENROUTER_MODEL,
        }
        for spec in OPENROUTER_MODELS
    ]
