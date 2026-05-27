from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import OcrConfig


def run_ocr(frames: list[dict[str, Any]], config: OcrConfig) -> dict[str, Any]:
    if not frames:
        return {"status": "skipped", "reason": "no_frames", "frames": []}

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "paddleocr_unavailable",
            "error": str(exc),
            "frames": _empty_ocr_frames(frames),
        }

    try:
        engine = PaddleOCR(use_angle_cls=True, lang=config.language)
        output_frames = []
        for frame in frames:
            image_path = frame["image_path"]
            result = engine.ocr(image_path, cls=True) or []
            texts = []
            for block in result:
                for line in block or []:
                    bbox, payload = line
                    text, confidence = payload
                    if confidence >= config.confidence_threshold:
                        texts.append({"text": text, "confidence": float(confidence), "bbox": bbox})
            raw_text = " ".join(item["text"] for item in texts)
            output_frames.append({**frame, "texts": texts, "raw_text": raw_text})
        return {"status": "ok", "frames": output_frames}
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "paddleocr_failed",
            "error": str(exc),
            "frames": _empty_ocr_frames(frames),
        }


def deduplicate_ocr(ocr_data: dict[str, Any], threshold: float) -> dict[str, Any]:
    previous_kept_text = ""
    previous_kept_id: str | None = None
    deduped = []
    for frame in ocr_data.get("frames", []):
        normalized = normalize_text(frame.get("raw_text", ""))
        duplicate = False
        duplicate_of = None
        if normalized and previous_kept_text:
            similarity = SequenceMatcher(None, previous_kept_text, normalized).ratio()
            duplicate = similarity >= threshold
            duplicate_of = previous_kept_id if duplicate else None
        frame_out = {
            **frame,
            "normalized_text": normalized,
            "is_duplicate": duplicate,
            "duplicate_of": duplicate_of,
        }
        deduped.append(frame_out)
        if normalized and not duplicate:
            previous_kept_text = normalized
            previous_kept_id = frame.get("frame_id")
    return {**ocr_data, "frames": deduped, "deduplicated": True, "duplicate_similarity_threshold": threshold}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s\r\n\t]+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _empty_ocr_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**frame, "texts": [], "raw_text": ""} for frame in frames]
