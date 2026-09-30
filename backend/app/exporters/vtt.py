"""WebVTT serializer. Pure serialization of quantized cues."""

from __future__ import annotations

from app.exporters.timestamps import format_vtt_timestamp
from app.subtitles.validate import QuantizedCue


def render(cues: list[QuantizedCue]) -> str:
    blocks: list[str] = []
    for cue in cues:
        start = format_vtt_timestamp(cue.start_ms)
        end = format_vtt_timestamp(cue.end_ms)
        blocks.append(f"{start} --> {end}\n{cue.text}")
    header = "WEBVTT"
    if not blocks:
        return header + "\n"
    return header + "\n\n" + "\n\n".join(blocks) + "\n"
