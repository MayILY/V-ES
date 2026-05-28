from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .io_utils import safe_stem


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    metadata: Path
    audio: Path
    transcript_json: Path
    transcript_srt: Path
    frames_dir: Path
    frames_metadata: Path
    ocr: Path
    scenes: Path
    scene_keyframes: Path
    frame_descriptions: Path
    timeline: Path
    summary_evidence_json: Path
    summary_evidence_md: Path
    timeline_summary: Path
    chapter_summaries: Path
    summary: Path
    run_log: Path


def build_output_paths(input_file: Path, output_dir: Path | None, config: Config) -> OutputPaths:
    root = output_dir or config.output.base_dir / safe_stem(input_file)
    return OutputPaths(
        root=root,
        metadata=root / "metadata.json",
        audio=root / "audio.wav",
        transcript_json=root / "transcript.json",
        transcript_srt=root / "transcript.srt",
        frames_dir=root / "frames",
        frames_metadata=root / "frames.json",
        ocr=root / "ocr.json",
        scenes=root / "scenes.json",
        scene_keyframes=root / "scene_keyframes.json",
        frame_descriptions=root / "frame_descriptions.json",
        timeline=root / "timeline_events.json",
        summary_evidence_json=root / "summary_evidence.json",
        summary_evidence_md=root / "summary_evidence.md",
        timeline_summary=root / "timeline_summary.md",
        chapter_summaries=root / "chapter_summaries.md",
        summary=root / "final_summary.md",
        run_log=root / "run.log",
    )


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")


def ensure_output_dir_available(root: Path, force: bool) -> None:
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {root}. Use --force to overwrite generated files.")
