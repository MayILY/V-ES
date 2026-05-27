from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import SceneDetectionConfig

SceneDetector = Callable[[Path, SceneDetectionConfig], list[dict[str, Any]]]


def detect_scenes(
    input_file: Path,
    metadata: dict[str, Any],
    config: SceneDetectionConfig,
    detector: SceneDetector | None = None,
) -> dict[str, Any]:
    if not metadata.get("has_video"):
        return {"status": "skipped", "reason": "no_video_stream", "scenes": []}

    if not config.enabled:
        return _fallback_whole_video(metadata, "scene_detection_disabled")

    try:
        scenes = (detector or _detect_with_pyscenedetect)(input_file, config)
    except ImportError as exc:
        return _fallback_whole_video(metadata, "pyscenedetect_unavailable", str(exc))
    except Exception as exc:
        return _fallback_whole_video(metadata, "pyscenedetect_failed", str(exc))

    if not scenes:
        return _fallback_whole_video(metadata, "no_scenes_detected")

    return {
        "status": "ok",
        "mode": "pyscenedetect",
        "threshold": config.threshold,
        "min_scene_len_sec": config.min_scene_len_sec,
        "scenes": _normalize_scenes(scenes, metadata, config.min_scene_len_sec),
    }


def _detect_with_pyscenedetect(input_file: Path, config: SceneDetectionConfig) -> list[dict[str, Any]]:
    try:
        from scenedetect import ContentDetector, detect  # type: ignore
    except Exception as exc:
        raise ImportError(exc) from exc

    detector = ContentDetector(threshold=config.threshold)
    raw_scenes = detect(str(input_file), detector)
    scenes = []
    for start, end in raw_scenes:
        scenes.append({"start": _timecode_seconds(start), "end": _timecode_seconds(end)})
    return scenes


def _fallback_whole_video(metadata: dict[str, Any], reason: str, error: str | None = None) -> dict[str, Any]:
    duration = float(metadata.get("duration_sec") or 0.0)
    scene = {
        "scene_id": 1,
        "start": 0.0,
        "end": round(duration, 3),
        "duration_sec": round(duration, 3),
    }
    result = {
        "status": "skipped",
        "reason": reason,
        "mode": "whole_video_fallback",
        "scenes": [scene] if duration > 0 else [],
    }
    if error:
        result["error"] = error
    return result


def _normalize_scenes(
    scenes: list[dict[str, Any]],
    metadata: dict[str, Any],
    min_scene_len_sec: float,
) -> list[dict[str, Any]]:
    duration = float(metadata.get("duration_sec") or 0.0)
    normalized = []
    for index, scene in enumerate(scenes, start=1):
        start = max(0.0, float(scene.get("start", 0.0)))
        end = float(scene.get("end", duration or start))
        if duration > 0:
            end = min(end, duration)
        if end <= start:
            continue
        if end - start < min_scene_len_sec:
            continue
        normalized.append(
            {
                "scene_id": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_sec": round(end - start, 3),
            }
        )
    return normalized


def _timecode_seconds(value: Any) -> float:
    if hasattr(value, "get_seconds"):
        return float(value.get_seconds())
    return float(value)
