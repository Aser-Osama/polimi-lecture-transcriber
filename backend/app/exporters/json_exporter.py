"""JSON serializer for the versioned transcript result schema."""

from __future__ import annotations

import json

from app.models.result import TranscriptResult


def render(result: TranscriptResult) -> str:
    payload = result.model_dump(mode="json")
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def parse(content: str) -> TranscriptResult:
    return TranscriptResult.model_validate(json.loads(content))
