from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from .config import VisionConfig


def describe_keyframes(scene_keyframes: dict[str, Any], config: VisionConfig) -> dict[str, Any]:
    frames = _flatten_keyframes(scene_keyframes)
    if not config.enabled:
        return {"status": "skipped", "reason": "vision_disabled", "frames": []}
    if not frames:
        return {"status": "skipped", "reason": "no_keyframes", "frames": []}
    if not os.environ.get("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "missing_openai_api_key", "frames": _empty_descriptions(frames)}

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "openai_package_unavailable",
            "error": str(exc),
            "frames": _empty_descriptions(frames),
        }

    client = OpenAI()
    described = []
    failures = 0
    for frame in frames[: config.max_frames]:
        try:
            described.append(_describe_frame(client, frame, config))
        except Exception as exc:
            failures += 1
            described.append({**frame, "status": "failed", "error": str(exc), "description": ""})

    status = "ok" if failures == 0 else "partial" if failures < len(described) else "failed"
    return {
        "status": status,
        "model": config.model,
        "max_frames": config.max_frames,
        "max_image_width": config.max_image_width,
        "detail": config.detail,
        "described_frame_count": len(described) - failures,
        "frames": described,
    }


def _describe_frame(client: Any, frame: dict[str, Any], config: VisionConfig) -> dict[str, Any]:
    data_url = _image_data_url(Path(str(frame["image_path"])), config.max_image_width)
    prompt = (
        "请只根据这一帧画面生成简洁中文描述。"
        "重点说明画面内容、可见 UI 或文字、可能的操作/场景。"
        "不要编造画面中没有出现的信息。"
    )
    response = client.responses.create(
        model=config.model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": config.detail},
                ],
            }
        ],
    )
    text = getattr(response, "output_text", None) or _extract_response_text(response)
    return {**frame, "status": "ok", "description": text.strip()}


def _image_data_url(path: Path, max_width: int) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    data = _resize_image_bytes(path, max_width) or path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _resize_image_bytes(path: Path, max_width: int) -> bytes | None:
    if max_width <= 0:
        return None
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            if image.width <= max_width:
                return None
            ratio = max_width / image.width
            size = (max_width, max(1, int(image.height * ratio)))
            resized = image.convert("RGB").resize(size)
            buffer = BytesIO()
            resized.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()
    except Exception:
        return None


def _flatten_keyframes(scene_keyframes: dict[str, Any]) -> list[dict[str, Any]]:
    frames = []
    seen_ids: set[str] = set()
    for scene in scene_keyframes.get("scenes", []):
        for frame in scene.get("keyframes", []):
            frame_id = str(frame.get("frame_id") or frame.get("image_path"))
            if frame_id in seen_ids:
                continue
            seen_ids.add(frame_id)
            frames.append({**frame, "scene_id": scene.get("scene_id")})
    return frames


def _empty_descriptions(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**frame, "status": "skipped", "description": ""} for frame in frames]


def _extract_response_text(response: Any) -> str:
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)
