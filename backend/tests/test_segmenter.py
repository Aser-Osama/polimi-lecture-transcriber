from __future__ import annotations

import math

from app.models.result import AlignedWord
from app.subtitles.segmenter import (
    SegmenterOptions,
    segment_from_segments,
    segment_words,
)


def make_words(
    spec: list[tuple[str, float, float]],
) -> list[AlignedWord]:
    return [
        AlignedWord(text=text, start=start, end=end, probability=0.9)
        for text, start, end in spec
    ]


def steady(
    texts: list[str],
    word_duration: float = 0.4,
    gap: float = 0.0,
    start: float = 0.0,
) -> list[AlignedWord]:
    words = []
    cursor = start
    for text in texts:
        words.append(AlignedWord(text=text, start=cursor, end=cursor + word_duration))
        cursor += word_duration + gap
    return words


DUMMY = ["word"] * 60


def test_empty_words():
    result = segment_words([])
    assert result.cues == []
    assert result.warnings == []


def test_single_sentence_no_limits_exceeded():
    words = steady(["Hello", "everyone", "and", "welcome."], word_duration=0.4, gap=0.1)
    result = segment_words(words)
    assert len(result.cues) == 1
    cue = result.cues[0]
    assert cue.start == 0.0
    assert words[-1].end <= cue.end <= words[-1].end + 0.3 + 1e-9
    assert cue.text == "Hello everyone and welcome."


def test_long_continuous_sentence_is_split_by_duration():
    words = steady(DUMMY, word_duration=0.5, gap=0.0)
    result = segment_words(words, SegmenterOptions(max_cue_duration=7.0))
    assert len(result.cues) >= 4
    for cue in result.cues:
        assert cue.end - cue.start <= 7.0 + 0.31  # padding bounded by 0.3
    # chronological and non-overlapping
    for previous, current in zip(result.cues, result.cues[1:]):
        assert previous.end <= current.start + 1e-9


def test_30_second_raw_segment_uses_real_word_times():
    words = steady(DUMMY, word_duration=0.6, gap=0.0)
    assert words[-1].end >= 30
    result = segment_words(words)
    assert len(result.cues) >= 4
    for cue in result.cues:
        first = words[cue.first_word_index]
        last = words[cue.last_word_index - 1]
        assert cue.start == round(first.start, 3)
        assert cue.end >= round(last.end, 3) - 1e-9
        assert cue.end - round(last.end, 3) <= 0.3 + 1e-9


def test_rapid_speech_respects_character_budget():
    words = steady(
        ["microarchitectural"] * 40, word_duration=0.15, gap=0.0
    )
    opts = SegmenterOptions(max_line_chars=42, max_lines=2)
    result = segment_words(words, opts)
    assert result.cues
    for cue in result.cues:
        for line in cue.lines:
            assert len(line) <= opts.max_line_chars + len("microarchitectural")
        assert len(cue.lines) <= opts.max_lines


def test_long_silence_forces_new_cue():
    words = [
        AlignedWord(text="Hello", start=0.0, end=0.4),
        AlignedWord(text="world.", start=0.5, end=0.9),
        AlignedWord(text="Goodbye", start=3.0, end=3.4),
        AlignedWord(text="world.", start=3.5, end=3.9),
    ]
    result = segment_words(words)
    assert len(result.cues) == 2
    assert result.cues[0].start == 0.0
    assert result.cues[1].start == 3.0


def test_tiny_silence_does_not_split():
    words = [
        AlignedWord(text="the", start=0.0, end=0.3),
        AlignedWord(text="kernel", start=0.4, end=0.8),
        AlignedWord(text="scheduler", start=0.9, end=1.4),
    ]
    result = segment_words(words)
    assert len(result.cues) == 1


def test_sentence_punctuation_breaks_when_readable():
    words = steady(
        ["The", "system", "boots.", "Then", "it", "runs."],
        word_duration=0.5,
        gap=0.15,
    )
    result = segment_words(words)
    assert len(result.cues) == 2
    assert result.cues[0].text.endswith("boots.")
    assert result.cues[1].text.startswith("Then")


def test_multiple_sentences_in_one_whisper_segment():
    words = steady(
        ["First", "sentence", "here!", "Second", "sentence.", "Third", "one?"],
        word_duration=0.5,
        gap=0.2,
    )
    result = segment_words(words)
    assert len(result.cues) == 3


def test_short_sentence_start_does_not_fragment():
    # "No. And..." spoken quickly: the first pseudo-sentence is too short and
    # must not create a flashing one-word cue.
    words = steady(["No.", "And", "yes."], word_duration=0.2, gap=0.0)
    result = segment_words(words)
    assert len(result.cues) == 1


def test_tiny_cue_merges_into_neighbor():
    words = [
        AlignedWord(text="moving", start=0.0, end=1.0),
        AlignedWord(text="along", start=1.1, end=2.0),
        AlignedWord(text="now", start=2.2, end=2.35),
        AlignedWord(text="finally", start=2.5, end=3.2),
    ]
    result = segment_words(words, SegmenterOptions(silence_break=10.0))
    durations = [c.end - c.start for c in result.cues]
    assert min(durations) >= 0.7, durations


def test_unmergeable_single_word_kept_with_warning():
    words = [
        AlignedWord(text="Wait", start=10.0, end=10.2),
    ]
    result = segment_words(words)
    assert len(result.cues) == 1
    assert any("shorter than" in w for w in result.warnings)


def test_line_length_constraints():
    words = steady(
        ["abcdefghij"] * 16, word_duration=0.3, gap=0.0
    )
    opts = SegmenterOptions(max_line_chars=30, max_lines=2)
    result = segment_words(words, opts)
    assert result.cues
    for cue in result.cues:
        assert len(cue.lines) <= 2
        for line in cue.lines:
            assert len(line) <= 30


def test_final_word_near_media_end():
    words = steady(["closing", "the", "lecture", "now."], word_duration=0.5)
    media_duration = words[-1].end
    result = segment_words(words, media_duration=media_duration)
    cue = result.cues[-1]
    assert cue.end <= media_duration + 1e-9


def test_word_beyond_media_end_dropped():
    words = steady(["hello", "world"], word_duration=0.5)
    result = segment_words(words, media_duration=0.4)
    assert len(result.cues) == 1
    assert result.cues[0].text.startswith("hello")
    assert any("out-of-range" in w or "malformed" in w for w in result.warnings)


def test_malformed_and_overlapping_words_are_repaired():
    words = [
        AlignedWord(text="alpha", start=0.0, end=0.5),
        AlignedWord(text="beta", start=0.3, end=0.8),  # overlaps previous
        AlignedWord(text="gamma", start=float("nan"), end=1.2),  # dropped
        AlignedWord(text="delta", start=1.0, end=1.4),
        AlignedWord(text="", start=1.5, end=1.7),  # dropped
        AlignedWord(text="epsilon", start=-0.5, end=1.9),  # negative start
    ]
    result = segment_words(words)
    assert result.cues, "expected cues from the valid words"
    for previous, current in zip(result.cues, result.cues[1:]):
        assert previous.end <= current.start + 1e-9
    assert any("malformed" in w for w in result.warnings)
    assert any("out-of-range" in w for w in result.warnings)


def test_end_padding_never_overlaps_next_cue():
    words = [
        AlignedWord(text="first", start=0.0, end=0.5),
        AlignedWord(text="part.", start=2.5, end=3.0),
    ]
    result = segment_words(words, SegmenterOptions(silence_break=1.0))
    assert len(result.cues) == 2
    assert result.cues[0].end <= result.cues[1].start


def test_segment_fallback_without_words():
    from app.models.result import RawSegment

    segments = [
        RawSegment(id=0, start=0.0, end=4.0, text="Hello everyone."),
        RawSegment(id=1, start=4.5, end=12.0, text="This is a longer segment that should wrap nicely."),
    ]
    result = segment_from_segments(segments)
    assert len(result.cues) == 2
    assert result.cues[0].start == 0.0
    assert result.cues[1].start == 4.5
    assert any("segment timing" in w for w in result.warnings)


def test_probabilities_preserved_through_sanitize():
    words = [AlignedWord(text="hello", start=0.0, end=0.5, probability=0.42)]
    result = segment_words(words)
    assert result.cues[0].first_word_index == 0
    assert math.isclose(words[0].probability, 0.42)
