"""SubRip (SRT) serializer. Pure serialization of quantized cues."""

from __future__ import annotations

from app.exporters.timestamps import format_srt_timestamp
from app.subtitles.validate import QuantizedCue


def render(cues: list[QuantizedCue]) -> str:
    blocks: list[str] = []
    for cue in cues:
        start = format_srt_timestamp(cue.start_ms)
        end = format_srt_timestamp(cue.end_ms)
        blocks.append(f"{cue.index}\n{start} --> {end}\n{cue.text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"
