from video_summarizer.transcript import format_srt_time


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(65.432) == "00:01:05,432"
    assert format_srt_time(3661.001) == "01:01:01,001"
