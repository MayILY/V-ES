from pathlib import Path

from video_summarizer.config import Config
from video_summarizer.paths import build_output_paths


def test_default_output_path_uses_safe_stem():
    paths = build_output_paths(Path("video/时彧的抖音 - 抖音 ().mp4"), None, Config())
    assert paths.root.parts[0] == "outputs"
    assert paths.metadata.name == "metadata.json"
    assert " " not in paths.root.name
