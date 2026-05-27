from video_summarizer.ffmpeg_tools import _parse_fps


def test_parse_fps_fraction():
    assert _parse_fps("30000/1001") == 29.97003


def test_parse_fps_empty_values():
    assert _parse_fps("0/0") is None
    assert _parse_fps(None) is None
