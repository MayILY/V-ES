from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .doctor import format_doctor_report, has_blocking_missing, run_doctor
from .ffmpeg_tools import merge_audio_video, probe_media
from .io_utils import write_json
from .media_scan import scan_directory
from .ocr_prepare import prepare_ocr
from .pipeline import inspect_media, run_pipeline

app = typer.Typer(
    help="Local-first video evidence extraction and summarization CLI.",
    no_args_is_help=True,
)


@app.command()
def doctor(
    config: Optional[Path] = typer.Option(None, "--config", help="Optional config.yaml path."),
) -> None:
    """Check local dependencies for the stable CLI pipeline."""
    cfg = load_config(config)
    checks = run_doctor(config=cfg)
    typer.echo(format_doctor_report(checks))
    if has_blocking_missing(checks):
        raise typer.Exit(code=1)


@app.command("ocr-prepare")
def ocr_prepare(
    config: Optional[Path] = typer.Option(None, "--config", help="Optional config.yaml path."),
) -> None:
    """Prepare and smoke-test local PP-OCRv5 mobile OCR models."""
    cfg = load_config(config)
    result = prepare_ocr(cfg.ocr)
    for key, value in result.items():
        typer.echo(f"{key}: {value}")
    if result.get("status") != "ok":
        raise typer.Exit(code=1)


@app.command()
def inspect(
    input_file: Path = typer.Argument(..., exists=True, readable=True, help="Local media file to inspect."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional config.yaml path."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing generated files."),
) -> None:
    """Run ffprobe and write metadata.json."""
    cfg = load_config(config)
    result = inspect_media(input_file=input_file, output_dir=output, config=cfg, force=force)
    typer.echo(f"metadata: {result}")


@app.command()
def scan(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Directory to scan."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for media_scan.json."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing scan output directory."),
    max_delta: float = typer.Option(1.0, "--max-delta", help="Maximum duration difference, in seconds, for pair candidates."),
) -> None:
    """Scan a directory and suggest audio-only/video-only pair candidates."""
    outputs = scan_directory(input_dir=input_dir, output_dir=output, force=force, max_duration_delta_sec=max_delta)
    typer.echo("scan outputs:")
    for name, path in outputs.items():
        typer.echo(f"- {name}: {path}")


@app.command()
def merge(
    video_file: Path = typer.Argument(..., exists=True, readable=True, help="Video-only media file."),
    audio_file: Path = typer.Argument(..., exists=True, readable=True, help="Audio-only media file."),
    output: Path = typer.Option(..., "--output", "-o", help="Merged media output path."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing merged output."),
) -> None:
    """Merge one video stream and one audio stream into a local MP4."""
    video_meta = probe_media(video_file)
    audio_meta = probe_media(audio_file)
    if not video_meta.get("has_video"):
        raise typer.BadParameter(f"video_file has no video stream: {video_file}")
    if not audio_meta.get("has_audio"):
        raise typer.BadParameter(f"audio_file has no audio stream: {audio_file}")

    result = merge_audio_video(video_file=video_file, audio_file=audio_file, output_file=output, force=force)
    status_path = output.parent / "merge_status.json"
    write_json(
        status_path,
        {
            "merge": result,
            "video": {
                "path": str(video_file),
                "duration_sec": video_meta.get("duration_sec"),
                "video_codec": video_meta.get("video_codec"),
                "width": video_meta.get("width"),
                "height": video_meta.get("height"),
            },
            "audio": {
                "path": str(audio_file),
                "duration_sec": audio_meta.get("duration_sec"),
                "audio_codec": audio_meta.get("audio_codec"),
            },
            "duration_delta_sec": _duration_delta(video_meta, audio_meta),
        },
    )
    typer.echo(f"merged: {output}")
    typer.echo(f"status: {status_path}")


@app.command()
def run(
    input_file: Path = typer.Argument(..., exists=True, readable=True, help="Local media file to process."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional config.yaml path."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing generated files."),
    frame_interval: Optional[float] = typer.Option(None, "--frame-interval", help="Seconds between sampled frames."),
    scene_detect: bool = typer.Option(False, "--scene-detect", help="Use PySceneDetect when available, with fallback."),
    vision: bool = typer.Option(False, "--vision", help="Use configured vision provider on selected keyframes when available."),
    summary_provider: Optional[str] = typer.Option(None, "--summary-provider", help="LLM provider for final summary."),
    summary_model: Optional[str] = typer.Option(None, "--summary-model", help="LLM model for final summary."),
    vision_provider: Optional[str] = typer.Option(None, "--vision-provider", help="LLM provider for keyframe vision."),
    vision_model: Optional[str] = typer.Option(None, "--vision-model", help="LLM model for keyframe vision."),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Maximum selected keyframes to send to vision."),
    max_image_width: Optional[int] = typer.Option(None, "--max-image-width", help="Maximum image width for vision inputs."),
    skip_summary: bool = typer.Option(False, "--skip-summary", help="Skip final LLM summary generation for local smoke/debug runs."),
    no_llm_cache: bool = typer.Option(False, "--no-llm-cache", help="Disable LLM cache reads and writes."),
    refresh_llm_cache: bool = typer.Option(False, "--refresh-llm-cache", help="Ignore existing LLM cache and write fresh successful outputs."),
) -> None:
    """Run the MVP pipeline for one local media file."""
    if no_llm_cache and refresh_llm_cache:
        raise typer.BadParameter("--no-llm-cache and --refresh-llm-cache are mutually exclusive.")
    cfg = load_config(config)
    if frame_interval is not None:
        cfg.ffmpeg.frame_interval_sec = frame_interval
    if scene_detect:
        cfg.scene_detection.enabled = True
    if vision:
        cfg.vision.enabled = True
    if summary_provider is not None:
        cfg.summary.provider = summary_provider
    if summary_model is not None:
        cfg.summary.model = summary_model
    if vision_provider is not None:
        cfg.vision.provider = vision_provider
    if vision_model is not None:
        cfg.vision.model = vision_model
    if max_frames is not None:
        cfg.vision.max_frames = max_frames
    if max_image_width is not None:
        cfg.vision.max_image_width = max_image_width
    if no_llm_cache:
        cfg.llm_cache.mode = "off"
    if refresh_llm_cache:
        cfg.llm_cache.mode = "refresh"
    outputs = run_pipeline(
        input_file=input_file,
        output_dir=output,
        config=cfg,
        force=force,
        skip_summary=skip_summary,
    )
    typer.echo("pipeline outputs:")
    for name, path in outputs.items():
        typer.echo(f"- {name}: {path}")
    run_status = json.loads(outputs["run_status"].read_text(encoding="utf-8"))
    if not skip_summary and run_status.get("summary", {}).get("status") == "failed":
        raise typer.Exit(code=1)


def _duration_delta(left: dict, right: dict) -> float | None:
    if left.get("duration_sec") is None or right.get("duration_sec") is None:
        return None
    return round(abs(float(left["duration_sec"]) - float(right["duration_sec"])), 3)
