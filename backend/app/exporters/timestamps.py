"""Timestamp formatting for subtitle formats (unit-tested)."""

from __future__ import annotations


def _split(ms: int) -> tuple[int, int, int, int]:
    if ms < 0:
        ms = 0
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return hours, minutes, seconds, millis


def format_srt_timestamp(ms: int) -> str:
    hours, minutes, seconds, millis = _split(ms)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_vtt_timestamp(ms: int) -> str:
    hours, minutes, seconds, millis = _split(ms)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
