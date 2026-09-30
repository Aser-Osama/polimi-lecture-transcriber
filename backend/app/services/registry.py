"""Whisper model catalog.

This is the single place where user-friendly quality tiers map to concrete
Hugging Face repository identifiers. All repo IDs were verified to exist on
the Hugging Face Hub (mlx-community), sizes are approximate download sizes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    repo_id: str
    approx_size_bytes: int
    description: str


MODEL_CATALOG: dict[str, ModelSpec] = {
    "quality": ModelSpec(
        key="quality",
        display_name="Quality - Whisper Large V3",
        repo_id="mlx-community/whisper-large-v3-mlx",
        approx_size_bytes=3_083_522_487,
        description="Highest-quality local transcription for lectures. Default.",
    ),
    "fast": ModelSpec(
        key="fast",
        display_name="Fast - Whisper Large V3 Turbo",
        repo_id="mlx-community/whisper-large-v3-turbo",
        approx_size_bytes=1_613_979_758,
        description="Much faster than Large V3 with a modest accuracy tradeoff.",
    ),
    "balanced": ModelSpec(
        key="balanced",
        display_name="Balanced - Whisper Medium",
        repo_id="mlx-community/whisper-medium-mlx",
        approx_size_bytes=1_524_927_044,
        description="Medium-sized model: lighter and quicker, less accurate than Large V3.",
    ),
}

DEFAULT_MODEL_KEY = "quality"


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[key]
    except KeyError:
        raise KeyError(f"Unknown model key: {key}") from None


def model_choices() -> list[dict]:
    """Serializable catalog for the API."""
    return [
        {
            "key": spec.key,
            "display_name": spec.display_name,
            "repo_id": spec.repo_id,
            "approx_size_bytes": spec.approx_size_bytes,
            "description": spec.description,
            "default": spec.key == DEFAULT_MODEL_KEY,
        }
        for spec in MODEL_CATALOG.values()
    ]
