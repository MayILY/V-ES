from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from .config import SceneDetectionConfig

FingerprintFn = Callable[[dict[str, Any]], str | None]


def select_scene_keyframes(
    scenes_result: dict[str, Any],
    frames: list[dict[str, Any]],
    config: SceneDetectionConfig,
    fingerprint_fn: FingerprintFn | None = None,
) -> dict[str, Any]:
    if not scenes_result.get("scenes"):
        return {"status": "skipped", "reason": "no_scenes", "scenes": [], "selected_frame_count": 0}
    if not frames:
        return {"status": "skipped", "reason": "no_frames", "scenes": [], "selected_frame_count": 0}

    fingerprint_fn = fingerprint_fn or frame_fingerprint
    seen_fingerprints: list[str] = []
    output_scenes = []

    for scene in scenes_result.get("scenes", []):
        candidates = _frames_in_scene(frames, scene)
        selected = []
        skipped_duplicates = []
        for frame in _pick_representative_frames(candidates, scene, config.max_keyframes_per_scene):
            fingerprint = fingerprint_fn(frame)
            if fingerprint and _is_duplicate(fingerprint, seen_fingerprints, config.duplicate_similarity_threshold):
                skipped_duplicates.append({**frame, "duplicate_reason": "similar_to_previous_keyframe"})
                continue
            if fingerprint:
                seen_fingerprints.append(fingerprint)
            selected.append(frame)
        output_scenes.append(
            {
                "scene_id": scene.get("scene_id"),
                "start": scene.get("start"),
                "end": scene.get("end"),
                "keyframes": selected,
                "skipped_duplicates": skipped_duplicates,
            }
        )

    return {
        "status": "ok",
        "source_scene_status": scenes_result.get("status"),
        "max_keyframes_per_scene": config.max_keyframes_per_scene,
        "duplicate_similarity_threshold": config.duplicate_similarity_threshold,
        "selected_frame_count": sum(len(scene["keyframes"]) for scene in output_scenes),
        "scenes": output_scenes,
    }


def frame_fingerprint(frame: dict[str, Any]) -> str | None:
    image_path = frame.get("image_path")
    if not image_path:
        return None
    path = Path(str(image_path))
    if not path.exists():
        return None

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((8, 8)).getdata())
        average = sum(pixels) / len(pixels)
        bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
        return f"ahash:{bits}"
    except Exception:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"sha256:{digest}"


def _frames_in_scene(frames: list[dict[str, Any]], scene: dict[str, Any]) -> list[dict[str, Any]]:
    start = float(scene.get("start", 0.0))
    end = float(scene.get("end", start))
    return [frame for frame in frames if start <= float(frame.get("timestamp", 0.0)) <= end]


def _pick_representative_frames(
    frames: list[dict[str, Any]],
    scene: dict[str, Any],
    max_count: int,
) -> list[dict[str, Any]]:
    if max_count <= 0 or not frames:
        return []
    targets = _targets_for_scene(scene, max_count)
    picked: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for target in targets:
        nearest = min(frames, key=lambda frame: abs(float(frame.get("timestamp", 0.0)) - target))
        frame_id = str(nearest.get("frame_id") or nearest.get("image_path"))
        if frame_id not in used_ids:
            picked.append(nearest)
            used_ids.add(frame_id)
    return picked[:max_count]


def _targets_for_scene(scene: dict[str, Any], max_count: int) -> list[float]:
    start = float(scene.get("start", 0.0))
    end = float(scene.get("end", start))
    midpoint = start + ((end - start) / 2)
    if max_count == 1:
        return [midpoint]
    if max_count == 2:
        return [start, end]
    return [start, midpoint, end]


def _is_duplicate(fingerprint: str, seen: list[str], threshold: float) -> bool:
    for previous in seen:
        if fingerprint == previous:
            return True
        if fingerprint.startswith("ahash:") and previous.startswith("ahash:"):
            if _hash_similarity(fingerprint[6:], previous[6:]) >= threshold:
                return True
    return False


def _hash_similarity(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / len(left)
