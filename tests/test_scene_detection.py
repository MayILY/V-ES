from pathlib import Path

from video_summarizer.config import SceneDetectionConfig
from video_summarizer.scene import create_scene_detector, detect_scenes, min_scene_len_frames


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
    def failing_detector(path, metadata, config):
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
    def fake_detector(path, metadata, config):
        return [{"start": 0, "end": 3.3333}, {"start": 3.3333, "end": 8}]

    result = detect_scenes(
        Path("sample.mp4"),
        {"has_video": True, "duration_sec": 8},
        SceneDetectionConfig(enabled=True),
        detector=fake_detector,
    )

    assert result["status"] == "ok"
    assert result["detector"] == "content"
    assert result["scenes"][0] == {"scene_id": 1, "start": 0.0, "end": 3.333, "duration_sec": 3.333}
    assert result["scenes"][1]["scene_id"] == 2


def test_min_scene_len_seconds_converts_to_frames_from_fps():
    config = SceneDetectionConfig(min_scene_len_sec=2.0)

    assert min_scene_len_frames({"fps": 30.0}, config) == 60


def test_create_content_detector_uses_threshold_and_min_scene_len():
    captured = {}

    class FakeContentDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeAdaptiveDetector:
        pass

    detector = create_scene_detector(
        FakeContentDetector,
        FakeAdaptiveDetector,
        {"fps": 30.0},
        SceneDetectionConfig(detector="content", threshold=31.0, min_scene_len_sec=2.0),
    )

    assert isinstance(detector, FakeContentDetector)
    assert captured == {"threshold": 31.0, "min_scene_len": 60}


def test_create_adaptive_detector_uses_adaptive_parameters():
    captured = {}

    class FakeContentDetector:
        pass

    class FakeAdaptiveDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    detector = create_scene_detector(
        FakeContentDetector,
        FakeAdaptiveDetector,
        {"fps": 24.0},
        SceneDetectionConfig(
            detector="adaptive",
            min_scene_len_sec=2.5,
            adaptive_threshold=4.0,
            window_width=3,
            min_content_val=12.0,
        ),
    )

    assert isinstance(detector, FakeAdaptiveDetector)
    assert captured == {
        "adaptive_threshold": 4.0,
        "min_scene_len": 60,
        "window_width": 3,
        "min_content_val": 12.0,
    }


def test_create_scene_detector_rejects_unknown_detector():
    class FakeContentDetector:
        pass

    class FakeAdaptiveDetector:
        pass

    try:
        create_scene_detector(
            FakeContentDetector,
            FakeAdaptiveDetector,
            {"fps": 30},
            SceneDetectionConfig(detector="unknown"),
        )
    except ValueError as exc:
        assert "Unsupported scene detector" in str(exc)
    else:
        raise AssertionError("Expected unsupported detector to raise ValueError")
