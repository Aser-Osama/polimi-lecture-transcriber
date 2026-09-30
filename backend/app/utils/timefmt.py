"""Duration formatting helpers."""

from __future__ import annotations


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_realtime_factor(media_seconds: float | None, processing_seconds: float | None) -> str:
    if not media_seconds or not processing_seconds or processing_seconds <= 0:
        return "--"
    return f"{media_seconds / processing_seconds:.1f}x realtime"
