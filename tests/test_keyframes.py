from video_summarizer.config import SceneDetectionConfig
from video_summarizer.keyframes import select_scene_keyframes


def test_keyframes_selects_at_most_three_per_scene():
    frames = [
        {"frame_id": "f0", "timestamp": 0, "image_path": "f0.jpg"},
        {"frame_id": "f1", "timestamp": 5, "image_path": "f1.jpg"},
        {"frame_id": "f2", "timestamp": 10, "image_path": "f2.jpg"},
        {"frame_id": "f3", "timestamp": 15, "image_path": "f3.jpg"},
    ]
    scenes = {"status": "ok", "scenes": [{"scene_id": 1, "start": 0, "end": 15}]}

    result = select_scene_keyframes(
        scenes,
        frames,
        SceneDetectionConfig(max_keyframes_per_scene=3),
        fingerprint_fn=lambda frame: frame["frame_id"],
    )

    assert result["status"] == "ok"
    assert result["selected_frame_count"] == 3
    assert [frame["frame_id"] for frame in result["scenes"][0]["keyframes"]] == ["f0", "f1", "f3"]


def test_keyframes_skips_duplicate_fingerprints():
    frames = [
        {"frame_id": "f0", "timestamp": 0, "image_path": "f0.jpg"},
        {"frame_id": "f1", "timestamp": 5, "image_path": "f1.jpg"},
        {"frame_id": "f2", "timestamp": 10, "image_path": "f2.jpg"},
    ]
    scenes = {"status": "ok", "scenes": [{"scene_id": 1, "start": 0, "end": 10}]}

    result = select_scene_keyframes(
        scenes,
        frames,
        SceneDetectionConfig(max_keyframes_per_scene=3),
        fingerprint_fn=lambda frame: "same" if frame["frame_id"] in {"f0", "f1"} else frame["frame_id"],
    )

    assert [frame["frame_id"] for frame in result["scenes"][0]["keyframes"]] == ["f0", "f2"]
    assert result["scenes"][0]["skipped_duplicates"][0]["frame_id"] == "f1"


def test_keyframes_handles_missing_sources():
    assert select_scene_keyframes({"scenes": []}, [], SceneDetectionConfig())["reason"] == "no_scenes"
    assert (
        select_scene_keyframes({"scenes": [{"scene_id": 1, "start": 0, "end": 1}]}, [], SceneDetectionConfig())[
            "reason"
        ]
        == "no_frames"
    )
