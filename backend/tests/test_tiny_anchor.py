from __future__ import annotations

from app.alignment.tiny_anchor import (
    build_alignment_windows,
    map_remote_tokens,
    normalize_token,
)
from app.models.result import AlignedWord, RawSegment


def anchors() -> list[AlignedWord]:
    return [
        AlignedWord(text="Welcome", start=0.0, end=0.5, segment_id=0),
        AlignedWord(text="to", start=0.5, end=0.7, segment_id=0),
        AlignedWord(text="the", start=0.7, end=0.9, segment_id=0),
        AlignedWord(text="lecture.", start=1.4, end=1.9, segment_id=1),
        AlignedWord(text="Today", start=2.2, end=2.5, segment_id=1),
        AlignedWord(text="we", start=2.6, end=2.7, segment_id=1),
        AlignedWord(text="begin.", start=2.8, end=3.2, segment_id=1),
    ]


def test_normalize_token_strips_punctuation_and_case():
    assert normalize_token("Lecture,") == "lecture"
    assert normalize_token("TLB.") == "tlb"
    assert normalize_token("multi-core") == "multicore"


def test_matched_tokens_keep_exact_anchor_times():
    remote = "Welcome to the lecture. Today we begin."
    mapping = map_remote_tokens(remote, anchors())
    assert mapping.ok
    assert mapping.matched_ratio == 1.0
    assert [(w.text, w.start) for w in mapping.words][:4] == [
        ("Welcome", 0.0),
        ("to", 0.5),
        ("the", 0.7),
        ("lecture.", 1.4),
    ]


def test_unmatched_tokens_interpolate_between_anchors():
    remote = "Welcome something longer here lecture. Today"
    mapping = map_remote_tokens(remote, anchors())
    assert mapping.ok
    # "something longer here" sits between Welcome (ends 0.5) and lecture. (starts 1.4)
    middle = [w for w in mapping.words if w.text in {"something", "longer", "here"}]
    assert middle
    assert all(w.start >= 0.5 - 1e-6 and w.end <= 1.4 + 1e-6 for w in middle)
    for previous, current in zip(middle, middle[1:], strict=False):
        assert previous.end <= current.start + 1e-6


def test_trailing_tokens_extend_after_last_anchor():
    remote = "Welcome to the lecture. Today we begin. Extra words here"
    mapping = map_remote_tokens(remote, anchors())
    assert mapping.ok
    trailing = [w for w in mapping.words if w.text in {"Extra", "words", "here"}]
    assert trailing
    assert trailing[0].start >= 3.2


def test_low_match_returns_not_ok():
    mapping = map_remote_tokens("completely different content entirely unrelated", anchors())
    assert not mapping.ok
    assert any("did not match" in warning for warning in mapping.warnings)


def test_empty_inputs():
    assert not map_remote_tokens("", anchors()).ok
    assert not map_remote_tokens("hello", []).ok


def test_media_duration_clamps_extrapolation():
    mapping = map_remote_tokens("Welcome lecture. trailing trailing trailing", anchors(), media_duration=2.0)
    assert mapping.ok
    assert all(word.start < 2.0 for word in mapping.words)


def test_build_alignment_windows_groups_by_anchor_segment():
    mapping = map_remote_tokens("Welcome to the lecture. Today we begin.", anchors())
    segments = [
        RawSegment(id=0, start=0.0, end=1.0, text="Welcome to the"),
        RawSegment(id=1, start=1.3, end=3.3, text="lecture. Today we begin."),
    ]
    windows = build_alignment_windows(mapping.words, segments)
    assert len(windows) == 2
    assert windows[0]["text"] == "Welcome to the"
    assert windows[1]["text"] == "lecture. Today we begin."
    assert windows[1]["start"] == 1.3
