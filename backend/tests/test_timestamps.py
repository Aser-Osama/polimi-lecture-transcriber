from app.exporters.timestamps import format_srt_timestamp, format_vtt_timestamp


def test_zero():
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_vtt_timestamp(0) == "00:00:00.000"


def test_milliseconds():
    assert format_srt_timestamp(1) == "00:00:00,001"
    assert format_srt_timestamp(999) == "00:00:00,999"


def test_seconds_and_minutes():
    assert format_srt_timestamp(61_500) == "00:01:01,500"
    assert format_vtt_timestamp(61_500) == "00:01:01.500"


def test_over_an_hour():
    assert format_srt_timestamp(3_661_001) == "01:01:01,001"
    assert format_vtt_timestamp(7_199_999) == "01:59:59.999"


def test_negative_is_clamped():
    assert format_srt_timestamp(-5) == "00:00:00,000"
    assert format_vtt_timestamp(-1000) == "00:00:00.000"


def test_long_lecture_timestamp():
    assert format_srt_timestamp(6_120_000) == "01:42:00,000"
