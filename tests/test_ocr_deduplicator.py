import os

from video_summarizer.config import OcrConfig
from video_summarizer.ocr import create_paddleocr_engine, deduplicate_ocr, normalize_text, parse_paddleocr_prediction


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


def test_parse_paddleocr_v3_prediction_filters_low_confidence():
    prediction = [
        {
            "res": {
                "rec_texts": ["系统架构", "低置信"],
                "rec_scores": [0.98, 0.2],
                "rec_polys": [
                    [[0, 0], [100, 0], [100, 30], [0, 30]],
                    [[0, 40], [100, 40], [100, 70], [0, 70]],
                ],
            }
        }
    ]

    texts = parse_paddleocr_prediction(prediction, confidence_threshold=0.5)

    assert texts == [
        {
            "text": "系统架构",
            "confidence": 0.98,
            "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]],
        }
    ]


def test_create_paddleocr_engine_uses_ppocrv5_mobile_names(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    monkeypatch.delenv("PADDLE_PDX_MODEL_SOURCE", raising=False)

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = OcrConfig(model_root=tmp_path)

    create_paddleocr_engine(FakePaddleOCR, config)

    assert captured["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert captured["text_recognition_model_name"] == "PP-OCRv5_mobile_rec"
    assert captured["device"] == "cpu"
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_doc_unwarping"] is False
    assert captured["use_textline_orientation"] is False
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(tmp_path)
    assert os.environ["PADDLE_PDX_MODEL_SOURCE"] == "BOS"
