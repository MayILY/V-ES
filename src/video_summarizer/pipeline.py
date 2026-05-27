from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .ffmpeg_tools import extract_audio, extract_frames, probe_media
from .io_utils import write_json
from .ocr import deduplicate_ocr, run_ocr
from .paths import build_output_paths, ensure_can_write, ensure_output_dir_available
from .summarize import summarize_timeline
from .timeline import build_timeline
from .transcript import empty_transcript, transcribe_audio, write_srt


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
    )
    write_json(paths.timeline, timeline)

    summary_status = summarize_timeline(timeline, paths.summary, config.summary, skip=skip_summary)
    write_json(paths.root / "run_status.json", {
        "metadata": "ok",
        "audio": audio_result,
        "transcript": {"status": transcript.get("status")},
        "frames": {"status": frames_result.get("status"), "frame_count": len(frames)},
        "ocr": {"status": ocr_result.get("status"), "reason": ocr_result.get("reason")},
        "summary": summary_status,
    })

    return {
        "metadata": paths.metadata,
        "transcript_json": paths.transcript_json,
        "transcript_srt": paths.transcript_srt,
        "frames_metadata": paths.frames_metadata,
        "ocr": paths.ocr,
        "timeline": paths.timeline,
        "summary": paths.summary,
    }
