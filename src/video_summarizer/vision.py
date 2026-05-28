from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from .config import LlmCacheConfig, LlmConfig, VisionConfig
from .llm import LlmClient, LlmProviderError, check_provider_ready, create_llm_client
from .llm_cache import cache_key, hash_bytes, lookup_text_cache, provider_cache_params, write_text_cache


def describe_keyframes(
    scene_keyframes: dict[str, Any],
    config: VisionConfig,
    llm_config: LlmConfig | None = None,
    llm_cache_config: LlmCacheConfig | None = None,
    llm_client: LlmClient | None = None,
) -> dict[str, Any]:
    frames = _flatten_keyframes(scene_keyframes)
    provider_status = {"provider": config.provider, "model": config.model}
    cache_config = llm_cache_config or LlmCacheConfig(mode="off")
    base_status = {
        "enabled": config.enabled,
        "called": False,
        "call_count": 0,
        "cache_mode": cache_config.mode,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "cache_write_count": 0,
    }
    if not config.enabled:
        return {"status": "skipped", "reason": "vision_disabled", "frames": [], **provider_status, **base_status}
    if not frames:
        return {"status": "skipped", "reason": "no_keyframes", "frames": [], **provider_status, **base_status}

    llm_config = llm_config or LlmConfig()
    provider_config = llm_config.providers.get(config.provider.lower().strip())
    client: LlmClient | None = llm_client
    described = []
    failures = 0
    call_count = 0
    cache_hit_count = 0
    cache_miss_count = 0
    cache_write_count = 0
    selected_frames = frames[: config.max_frames]

    for index, frame in enumerate(selected_frames):
        prompt = _vision_prompt()
        image_path = Path(str(frame["image_path"]))
        image_data = _resize_image_bytes(image_path, config.max_image_width)
        payload = image_data if image_data is not None else _read_image_bytes(image_path)
        key_payload = {
            "cache_schema_version": cache_config.schema_version,
            "provider": config.provider,
            "model": config.model,
            "prompt_template_version": config.prompt_template_version,
            "image_preprocessing_version": config.image_preprocessing_version,
            "detail": config.detail,
            "max_image_width": config.max_image_width,
            "provider_params": provider_cache_params(provider_config),
            "image_hash": hash_bytes(payload),
        }
        key = cache_key(key_payload)
        lookup = lookup_text_cache(cache_config, "vision", key)
        if lookup.hit:
            cache_hit_count += 1
            described.append({**frame, "status": "ok", "description": lookup.text or "", "cache_hit": True, "cache_key": key})
            continue
        if cache_config.mode != "off":
            cache_miss_count += 1
        if client is None:
            readiness = check_provider_ready(config.provider, config.model, llm_config, need_vision=True)
            if readiness.status != "ok":
                return {
                    "status": "skipped",
                    "reason": readiness.reason or "provider_vision_unavailable",
                    "frames": described + _empty_descriptions(selected_frames[index:]),
                    **provider_status,
                    **base_status,
                    "cache_hit_count": cache_hit_count,
                    "cache_miss_count": cache_miss_count,
                }
            try:
                client = create_llm_client(config.provider, config.model, llm_config)
            except LlmProviderError as exc:
                return {
                    "status": "skipped",
                    "reason": str(exc),
                    "frames": described + _empty_descriptions(selected_frames[index:]),
                    **provider_status,
                    **base_status,
                    "cache_hit_count": cache_hit_count,
                    "cache_miss_count": cache_miss_count,
                }
        try:
            text = client.describe_image(image_path, prompt, detail=config.detail, image_data=image_data)
            call_count += 1
            cache_path = write_text_cache(cache_config, "vision", key, text.strip(), key_payload)
            if cache_path is not None:
                cache_write_count += 1
            described.append({**frame, "status": "ok", "description": text.strip(), "cache_hit": False, "cache_key": key})
        except Exception as exc:
            failures += 1
            described.append({**frame, "status": "failed", "error": str(exc), "description": ""})

    status = "ok" if failures == 0 else "partial" if failures < len(described) else "failed"
    return {
        "status": status,
        **provider_status,
        "max_frames": config.max_frames,
        "max_image_width": config.max_image_width,
        "detail": config.detail,
        "described_frame_count": len(described) - failures,
        "frames": described,
        "enabled": config.enabled,
        "called": call_count > 0,
        "call_count": call_count,
        "cache_mode": cache_config.mode,
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "cache_write_count": cache_write_count,
    }


def _describe_frame(client: LlmClient, frame: dict[str, Any], config: VisionConfig) -> dict[str, Any]:
    image_path = Path(str(frame["image_path"]))
    image_data = _resize_image_bytes(image_path, config.max_image_width)
    text = client.describe_image(image_path, _vision_prompt(), detail=config.detail, image_data=image_data)
    return {**frame, "status": "ok", "description": text.strip()}


def _vision_prompt() -> str:
    return (
        "Describe this single video frame in concise Chinese. Focus only on visible content, UI/text, "
        "and likely operation or scene. Do not invent information not visible in the frame."
    )


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


def _read_image_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        return b""


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
