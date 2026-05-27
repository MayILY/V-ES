from video_summarizer.media_scan import find_candidate_pairs


def test_find_candidate_pairs_matches_duration_within_delta():
    files = [
        {
            "path": "video.mp4",
            "name": "video.mp4",
            "has_audio": False,
            "has_video": True,
            "duration_sec": 100.2,
        },
        {
            "path": "audio.mp4",
            "name": "audio.mp4",
            "has_audio": True,
            "has_video": False,
            "duration_sec": 100.8,
        },
        {
            "path": "far-audio.mp4",
            "name": "far-audio.mp4",
            "has_audio": True,
            "has_video": False,
            "duration_sec": 110.0,
        },
    ]

    pairs = find_candidate_pairs(files, max_duration_delta_sec=1.0)

    assert len(pairs) == 1
    assert pairs[0]["video_path"] == "video.mp4"
    assert pairs[0]["audio_path"] == "audio.mp4"
    assert pairs[0]["duration_delta_sec"] == 0.6


def test_find_candidate_pairs_ignores_complete_media_files():
    files = [
        {
            "path": "complete.mp4",
            "name": "complete.mp4",
            "has_audio": True,
            "has_video": True,
            "duration_sec": 100.0,
        },
        {
            "path": "audio.mp4",
            "name": "audio.mp4",
            "has_audio": True,
            "has_video": False,
            "duration_sec": 100.0,
        },
    ]

    assert find_candidate_pairs(files, max_duration_delta_sec=1.0) == []
