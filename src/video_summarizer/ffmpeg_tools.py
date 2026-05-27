from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import FfmpegConfig


def probe_media(input_file: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels,duration:stream_tags=language",
        "-of",
        "json",
        str(input_file),
    ]
    raw = _run(command)
    data = json.loads(raw.stdout)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    subtitle_stream = next((s for s in streams if s.get("codec_type") == "subtitle"), None)
    duration = _float_or_none(data.get("format", {}).get("duration"))
    fps = _parse_fps(video_stream.get("avg_frame_rate")) if video_stream else None

    return {
        "video_path": str(input_file),
        "duration_sec": duration,
        "size_bytes": _int_or_none(data.get("format", {}).get("size")),
        "has_audio": audio_stream is not None,
        "has_video": video_stream is not None,
        "has_subtitle": subtitle_stream is not None,
        "video_codec": video_stream.get("codec_name") if video_stream else None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "width": video_stream.get("width") if video_stream else None,
        "height": video_stream.get("height") if video_stream else None,
        "fps": fps,
        "streams": streams,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_audio(input_file: Path, output_path: Path, config: FfmpegConfig, force: bool) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y" if force else "-n",
        "-i",
        str(input_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.audio_sample_rate),
        str(output_path),
    ]
    result = _run(command, allow_failure=True)
    if result.returncode != 0:
        return {
            "status": "failed",
            "path": str(output_path),
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    return {"status": "ok", "path": str(output_path)}


def extract_frames(
    input_file: Path,
    frames_dir: Path,
    duration_sec: float | None,
    config: FfmpegConfig,
    force: bool,
) -> dict[str, Any]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for stale_frame in frames_dir.glob("frame_*.jpg"):
            stale_frame.unlink()
    pattern = frames_dir / "frame_%06d.jpg"
    scale_filter = f"fps=1/{config.frame_interval_sec},scale='min({config.max_frame_width},iw)':-2"
    command = [
        "ffmpeg",
        "-y" if force else "-n",
        "-i",
        str(input_file),
        "-vf",
        scale_filter,
        "-q:v",
        "3",
        str(pattern),
    ]
    result = _run(command, allow_failure=True)
    if result.returncode != 0:
        return {
            "status": "failed",
            "frames": [],
            "error": result.stderr.strip() or result.stdout.strip(),
        }

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    frames = []
    for index, frame_path in enumerate(frame_paths, start=1):
        timestamp = (index - 1) * config.frame_interval_sec
        if duration_sec is not None:
            timestamp = min(timestamp, max(duration_sec, 0.0))
        frames.append(
            {
                "frame_id": frame_path.stem,
                "timestamp": timestamp,
                "image_path": frame_path.as_posix(),
            }
        )
    return {"status": "ok", "frames": frames, "frame_count": len(frames)}


def merge_audio_video(video_file: Path, audio_file: Path, output_file: Path, force: bool) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_file}. Use --force to overwrite.")

    command = [
        "ffmpeg",
        "-y" if force else "-n",
        "-i",
        str(video_file),
        "-i",
        str(audio_file),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_file),
    ]
    result = _run(command, allow_failure=True)
    if result.returncode != 0:
        return {
            "status": "failed",
            "path": str(output_file),
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    return {"status": "ok", "path": str(output_file)}


def _run(command: list[str], allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        den = float(denominator)
        if math.isclose(den, 0.0):
            return None
        return round(float(numerator) / den, 6)
    return _float_or_none(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
