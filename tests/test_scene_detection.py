from pathlib import Path

from video_summarizer.config import SceneDetectionConfig
from video_summarizer.scene import detect_scenes


def test_scene_detection_disabled_uses_whole_video_fallback():
    result = detect_scenes(
        Path("sample.mp4"),
        {"has_video": True, "duration_sec": 12.5},
        SceneDetectionConfig(enabled=False),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "scene_detection_disabled"
    assert result["scenes"] == [{"scene_id": 1, "start": 0.0, "end": 12.5, "duration_sec": 12.5}]


def test_scene_detection_no_video_skips_without_fallback_scene():
    result = detect_scenes(
        Path("sample.mp4"),
        {"has_video": False, "duration_sec": 12.5},
        SceneDetectionConfig(enabled=True),
    )

    assert result == {"status": "skipped", "reason": "no_video_stream", "scenes": []}


def test_scene_detection_failure_uses_whole_video_fallback():
    def failing_detector(path, config):
        raise RuntimeError("boom")

    result = detect_scenes(
        Path("sample.mp4"),
        {"has_video": True, "duration_sec": 20},
        SceneDetectionConfig(enabled=True),
        detector=failing_detector,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "pyscenedetect_failed"
    assert result["scenes"][0]["end"] == 20.0


def test_scene_detection_normalizes_detected_scenes():
    def fake_detector(path, config):
        return [{"start": 0, "end": 3.3333}, {"start": 3.3333, "end": 8}]

    result = detect_scenes(
        Path("sample.mp4"),
        {"has_video": True, "duration_sec": 8},
        SceneDetectionConfig(enabled=True),
        detector=fake_detector,
    )

    assert result["status"] == "ok"
    assert result["scenes"][0] == {"scene_id": 1, "start": 0.0, "end": 3.333, "duration_sec": 3.333}
    assert result["scenes"][1]["scene_id"] == 2
