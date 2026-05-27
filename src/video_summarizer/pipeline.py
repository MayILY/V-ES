from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .ffmpeg_tools import extract_audio, extract_frames, probe_media
from .io_utils import write_json
from .keyframes import select_scene_keyframes
from .ocr import deduplicate_ocr, run_ocr
from .paths import build_output_paths, ensure_can_write, ensure_output_dir_available
from .scene import detect_scenes
from .summarize import summarize_timeline
from .timeline import build_timeline
from .transcript import empty_transcript, transcribe_audio, write_srt
from .vision import describe_keyframes


def inspect_media(input_file: Path, output_dir: Path | None, config: Config, force: bool = False) -> Path:
    paths = build_output_paths(input_file, output_dir, config)
    ensure_can_write(paths.metadata, force)
    metadata = probe_media(input_file)
    write_json(paths.metadata, metadata)
    return paths.metadata


def run_pipeline(
    input_file: Path,
    output_dir: Path | None,
    config: Config,
    force: bool = False,
    skip_summary: bool = False,
) -> dict[str, Path]:
    paths = build_output_paths(input_file, output_dir, config)
    ensure_output_dir_available(paths.root, force)
    paths.root.mkdir(parents=True, exist_ok=True)

    metadata = probe_media(input_file)
    write_json(paths.metadata, metadata)

    audio_result: dict[str, Any]
    if metadata.get("has_audio"):
        audio_result = extract_audio(input_file, paths.audio, config.ffmpeg, force=force)
    else:
        audio_result = {"status": "skipped", "reason": "no_audio_stream", "path": str(paths.audio)}

    if audio_result.get("status") == "ok":
        transcript = transcribe_audio(paths.audio, config.whisper)
    else:
        transcript = empty_transcript(audio_result.get("status", "skipped"), audio_result.get("reason", "audio_unavailable"))
    write_json(paths.transcript_json, transcript)
    write_srt(paths.transcript_srt, transcript)

    if metadata.get("has_video"):
        frames_result = extract_frames(
            input_file,
            paths.frames_dir,
            metadata.get("duration_sec"),
            config.ffmpeg,
            force=force,
        )
    else:
        frames_result = {"status": "skipped", "reason": "no_video_stream", "frames": []}
    frames = frames_result.get("frames", [])
    write_json(paths.frames_metadata, frames_result)

    scenes_result = detect_scenes(input_file, metadata, config.scene_detection)
    write_json(paths.scenes, scenes_result)

    scene_keyframes = select_scene_keyframes(scenes_result, frames, config.scene_detection)
    write_json(paths.scene_keyframes, scene_keyframes)

    frame_descriptions = describe_keyframes(scene_keyframes, config.vision)
    write_json(paths.frame_descriptions, frame_descriptions)

    ocr_result = run_ocr(frames, config.ocr)
    if config.ocr.deduplicate:
        ocr_result = deduplicate_ocr(ocr_result, config.ocr.duplicate_similarity_threshold)
    write_json(paths.ocr, ocr_result)

    timeline = build_timeline(
        metadata=metadata,
        transcript=transcript,
        ocr=ocr_result,
        frames=frames,
        window_sec=config.summary.segment_window_sec,
        frame_descriptions=frame_descriptions,
    )
    write_json(paths.timeline, timeline)

    summary_status = summarize_timeline(
        timeline,
        paths.summary,
        config.summary,
        skip=skip_summary,
        timeline_summary_path=paths.timeline_summary,
        chapter_summaries_path=paths.chapter_summaries,
    )
    write_json(paths.root / "run_status.json", {
        "metadata": "ok",
        "audio": audio_result,
        "transcript": {"status": transcript.get("status")},
        "frames": {"status": frames_result.get("status"), "frame_count": len(frames)},
        "scenes": {"status": scenes_result.get("status"), "reason": scenes_result.get("reason")},
        "scene_keyframes": {
            "status": scene_keyframes.get("status"),
            "selected_frame_count": scene_keyframes.get("selected_frame_count"),
        },
        "vision": {
            "status": frame_descriptions.get("status"),
            "reason": frame_descriptions.get("reason"),
            "described_frame_count": frame_descriptions.get("described_frame_count"),
        },
        "ocr": {"status": ocr_result.get("status"), "reason": ocr_result.get("reason")},
        "summary": summary_status,
    })

    return {
        "metadata": paths.metadata,
        "transcript_json": paths.transcript_json,
        "transcript_srt": paths.transcript_srt,
        "frames_metadata": paths.frames_metadata,
        "scenes": paths.scenes,
        "scene_keyframes": paths.scene_keyframes,
        "frame_descriptions": paths.frame_descriptions,
        "ocr": paths.ocr,
        "timeline": paths.timeline,
        "timeline_summary": paths.timeline_summary,
        "chapter_summaries": paths.chapter_summaries,
        "summary": paths.summary,
    }
