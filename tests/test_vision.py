from video_summarizer.config import VisionConfig
from video_summarizer.vision import describe_keyframes


def test_vision_disabled_skips():
    result = describe_keyframes(
        {"scenes": [{"scene_id": 1, "keyframes": [{"frame_id": "f1", "image_path": "f1.jpg"}]}]},
        VisionConfig(enabled=False),
    )

    assert result == {"status": "skipped", "reason": "vision_disabled", "frames": []}


def test_vision_missing_key_skips_with_frame_placeholders(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = describe_keyframes(
        {"scenes": [{"scene_id": 1, "keyframes": [{"frame_id": "f1", "image_path": "f1.jpg"}]}]},
        VisionConfig(enabled=True),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "missing_openai_api_key"
    assert result["frames"][0]["status"] == "skipped"
