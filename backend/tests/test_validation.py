from __future__ import annotations

from app.models.result import AlignedWord, Cue
from app.subtitles.validate import (
    quantize_cues,
    validate_and_repair_cues,
)


def cue(index: int, start: float, end: float, text: str = "text") -> Cue:
    return Cue(
        index=index,
        start=start,
        end=end,
        lines=[text],
        text=text,
        first_word_index=0,
        last_word_index=0,
    )


def words_for(*specs) -> list[AlignedWord]:
    return [AlignedWord(text=t, start=s, end=e) for t, s, e in specs]


def test_valid_cues_pass_through():
    cues = [cue(1, 0.0, 1.0), cue(2, 1.5, 2.5)]
    result = validate_and_repair_cues(cues, [])
    assert [c.index for c in result.cues] == [1, 2]
    assert result.warnings == []


def test_negative_start_clamped():
    result = validate_and_repair_cues([cue(1, -2.0, 1.0)], [])
    assert result.cues[0].start == 0.0
    assert any("negative" in w for w in result.warnings)


def test_start_not_before_end_dropped():
    result = validate_and_repair_cues([cue(1, 2.0, 2.0)], [])
    assert result.cues == []
    assert any("start is not before end" in w for w in result.warnings)


def test_beyond_media_duration_clamped():
    result = validate_and_repair_cues([cue(1, 1.0, 99.0)], [], media_duration=10.0)
    assert result.cues[0].end == 10.0
    assert any("media duration" in w for w in result.warnings)


def test_empty_text_dropped():
    result = validate_and_repair_cues([cue(1, 0.0, 1.0, text="   ")], [])
    assert result.cues == []


def test_non_finite_dropped():
    result = validate_and_repair_cues([cue(1, float("nan"), 1.0)], [])
    assert result.cues == []


def test_overlap_is_trimmed():
    cues = [cue(1, 0.0, 2.0), cue(2, 1.5, 3.0)]
    result = validate_and_repair_cues(cues, [])
    assert len(result.cues) == 2
    assert result.cues[0].end < result.cues[1].start
    assert any("Trimmed" in w for w in result.warnings)


def test_full_overlap_is_merged():
    cues = [cue(1, 0.0, 2.0, "one"), cue(2, 0.0, 2.0, "two")]
    result = validate_and_repair_cues(cues, [])
    assert len(result.cues) == 1
    assert "one" in result.cues[0].text and "two" in result.cues[0].text
    assert any("Merged" in w for w in result.warnings)


def test_word_range_warning():
    cues = [cue(1, 0.0, 1.0)]
    cues[0].first_word_index = 0
    cues[0].last_word_index = 1
    words = words_for(("hello", 5.0, 6.0))
    result = validate_and_repair_cues(cues, words)
    assert any("outside the cue range" in w for w in result.warnings)


def test_quantize_rounds_to_ms():
    result, warnings = quantize_cues([cue(1, 1.0004, 2.9996)])
    assert result[0].start_ms == 1000
    assert result[0].end_ms == 3000
    assert warnings == []


def test_quantize_removes_rounding_overlap():
    cues = [cue(1, 0.0, 1.0), cue(2, 0.9994, 2.0)]
    result, warnings = quantize_cues(cues)
    assert result[0].end_ms <= result[1].start_ms
    assert warnings


def test_quantize_equal_boundary_is_not_an_overlap():
    cues = [cue(1, 0.0, 1.0), cue(2, 1.0004, 2.0)]
    result, warnings = quantize_cues(cues)
    assert result[0].end_ms == result[1].start_ms
    assert warnings == []


def test_quantize_collapses_degenerate_cue():
    cues = [cue(1, 0.0, 0.0001)]
    result, _ = quantize_cues(cues)
    assert result[0].end_ms > result[0].start_ms


def test_quantize_reindexes():
    cues = [cue(7, 0.0, 1.0), cue(9, 1.5, 2.0)]
    result, _ = quantize_cues(cues)
    assert [c.index for c in result] == [1, 2]
