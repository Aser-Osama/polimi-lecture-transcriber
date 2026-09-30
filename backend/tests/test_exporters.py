from __future__ import annotations

import re

from app.exporters import srt, txt, vtt
from app.models.result import RawSegment
from app.subtitles.validate import QuantizedCue


def cues() -> list[QuantizedCue]:
    return [
        QuantizedCue(index=1, start_ms=0, end_ms=1500, text="Héllo, wörld!"),
        QuantizedCue(index=2, start_ms=2500, end_ms=4000, text="Second line."),
    ]


def test_srt_structure():
    rendered = srt.render(cues())
    blocks = rendered.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].startswith("1\n00:00:00,000 --> 00:00:01,500\n")
    assert blocks[1].startswith("2\n00:00:02,500 --> 00:00:04,000\n")
    assert rendered.endswith("\n")


def test_srt_unicode_preserved():
    rendered = srt.render(cues())
    assert "Héllo, wörld!" in rendered


def test_srt_empty():
    assert srt.render([]) == ""


def test_vtt_structure():
    rendered = vtt.render(cues())
    assert rendered.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500" in rendered
    assert "00:00:02.500 --> 00:00:04.000" in rendered
    assert "\r" not in rendered


def test_vtt_empty_has_header():
    assert vtt.render([]) == "WEBVTT\n"


def test_srt_no_overlaps_after_render():
    rendered = srt.render(cues())
    stamps = re.findall(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)", rendered)

    def to_ms(value: str) -> int:
        h, m, rest = value.split(":")
        s, ms = rest.split(",")
        return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)

    for (_, end), (start, _) in zip(stamps, stamps[1:], strict=False):
        assert to_ms(end) <= to_ms(start)


def test_txt_cleanup_and_paragraphs():
    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="Hello   world ,  this is   fine."),
        RawSegment(id=1, start=1.2, end=2.0, text="Same paragraph continues."),
        RawSegment(id=2, start=5.0, end=6.0, text="New paragraph after silence."),
    ]
    rendered = txt.render(segments)
    assert "Hello world, this is fine. Same paragraph continues." in rendered
    assert "continues.\n\nNew paragraph" in rendered
    assert rendered.endswith("\n")


def test_txt_empty():
    assert txt.render([]) == ""
