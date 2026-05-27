from __future__ import annotations

from pathlib import Path
from typing import Any

from .ffmpeg_tools import probe_media
from .io_utils import write_json

MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def scan_directory(input_dir: Path, output_dir: Path, force: bool, max_duration_delta_sec: float = 1.0) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --force to overwrite scan files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        metadata = probe_media(path)
        files.append(
            {
                "path": str(path),
                "name": path.name,
                "duration_sec": metadata.get("duration_sec"),
                "has_audio": metadata.get("has_audio"),
                "has_video": metadata.get("has_video"),
                "audio_codec": metadata.get("audio_codec"),
                "video_codec": metadata.get("video_codec"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "fps": metadata.get("fps"),
                "size_bytes": metadata.get("size_bytes"),
            }
        )

    candidate_pairs = find_candidate_pairs(files, max_duration_delta_sec=max_duration_delta_sec)
    scan = {
        "input_dir": str(input_dir),
        "file_count": len(files),
        "max_duration_delta_sec": max_duration_delta_sec,
        "files": files,
        "candidate_pairs": candidate_pairs,
    }
    scan_path = write_json(output_dir / "media_scan.json", scan)
    pairs_path = write_json(output_dir / "candidate_pairs.json", {"candidate_pairs": candidate_pairs})
    return {"scan": scan_path, "pairs": pairs_path}


def find_candidate_pairs(files: list[dict[str, Any]], max_duration_delta_sec: float = 1.0) -> list[dict[str, Any]]:
    audio_only = [
        item
        for item in files
        if item.get("has_audio") and not item.get("has_video") and item.get("duration_sec") is not None
    ]
    video_only = [
        item
        for item in files
        if item.get("has_video") and not item.get("has_audio") and item.get("duration_sec") is not None
    ]

    pairs = []
    for video in video_only:
        for audio in audio_only:
            delta = abs(float(video["duration_sec"]) - float(audio["duration_sec"]))
            if delta <= max_duration_delta_sec:
                pairs.append(
                    {
                        "video_path": video["path"],
                        "audio_path": audio["path"],
                        "video_name": video["name"],
                        "audio_name": audio["name"],
                        "video_duration_sec": video["duration_sec"],
                        "audio_duration_sec": audio["duration_sec"],
                        "duration_delta_sec": round(delta, 3),
                    }
                )
    return sorted(pairs, key=lambda item: (item["duration_delta_sec"], item["video_name"], item["audio_name"]))
