from video_summarizer.timeline import build_timeline


def test_timeline_handles_missing_sources():
    timeline = build_timeline(
        metadata={"duration_sec": 65},
        transcript={"segments": []},
        ocr={"frames": []},
        frames=[],
        window_sec=30,
    )

    assert len(timeline["events"]) == 3
    assert timeline["events"][0]["transcript"] == ""
    assert timeline["events"][0]["ocr_text"] == ""


def test_timeline_merges_transcript_and_ocr_by_window():
    timeline = build_timeline(
        metadata={"duration_sec": 60},
        transcript={"segments": [{"id": 1, "start": 5, "end": 10, "text": "hello"}]},
        ocr={"frames": [{"frame_id": "f1", "timestamp": 6, "raw_text": "title", "is_duplicate": False}]},
        frames=[{"frame_id": "f1", "timestamp": 6, "image_path": "frames/f1.jpg"}],
        window_sec=30,
    )

    assert timeline["events"][0]["transcript"] == "hello"
    assert timeline["events"][0]["ocr_text"] == "title"
