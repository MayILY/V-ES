from video_summarizer.ocr import deduplicate_ocr, normalize_text


def test_normalize_text_removes_spacing_and_punctuation():
    assert normalize_text(" 系统 架构 / OCR！") == "系统架构ocr"


def test_deduplicate_ocr_marks_adjacent_duplicate():
    data = {
        "status": "ok",
        "frames": [
            {"frame_id": "frame_000001", "timestamp": 0, "raw_text": "系统架构 OCR"},
            {"frame_id": "frame_000002", "timestamp": 5, "raw_text": "系统 架构 OCR!"},
            {"frame_id": "frame_000003", "timestamp": 10, "raw_text": "新的页面"},
        ],
    }

    result = deduplicate_ocr(data, threshold=0.9)

    assert result["frames"][0]["is_duplicate"] is False
    assert result["frames"][1]["is_duplicate"] is True
    assert result["frames"][1]["duplicate_of"] == "frame_000001"
    assert result["frames"][2]["is_duplicate"] is False
