from __future__ import annotations

import re
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import OcrConfig


DETECTION_MODEL_DIRNAME = "PP-OCRv5_mobile_det"
RECOGNITION_MODEL_DIRNAME = "PP-OCRv5_mobile_rec"


def run_ocr(frames: list[dict[str, Any]], config: OcrConfig) -> dict[str, Any]:
    if not frames:
        return {"status": "skipped", "reason": "no_frames", "frames": []}

    ensure_paddleocr_model_env(config)
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
        engine = create_paddleocr_engine(PaddleOCR, config)
        output_frames = []
        for frame in frames:
            image_path = frame["image_path"]
            result = engine.predict(image_path) or []
            texts = parse_paddleocr_prediction(result, config.confidence_threshold)
            raw_text = " ".join(item["text"] for item in texts)
            output_frames.append({**frame, "texts": texts, "raw_text": raw_text})
        return {
            "status": "ok",
            "engine": config.engine,
            "version": config.version,
            "device": config.device,
            "text_detection_model_name": config.text_detection_model_name,
            "text_recognition_model_name": config.text_recognition_model_name,
            "model_root": str(config.model_root),
            "frames": output_frames,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "paddleocr_failed",
            "error": str(exc),
            "frames": _empty_ocr_frames(frames),
        }


def create_paddleocr_engine(paddleocr_cls: Any, config: OcrConfig) -> Any:
    ensure_paddleocr_model_env(config)
    kwargs = {
        "text_detection_model_name": config.text_detection_model_name,
        "text_recognition_model_name": config.text_recognition_model_name,
        "use_doc_orientation_classify": config.use_doc_orientation_classify,
        "use_doc_unwarping": config.use_doc_unwarping,
        "use_textline_orientation": config.use_textline_orientation,
        "device": config.device,
        "engine": "paddle",
    }
    detection_dir = text_detection_model_dir(config)
    recognition_dir = text_recognition_model_dir(config)
    if detection_dir.exists():
        kwargs["text_detection_model_dir"] = str(detection_dir)
    if recognition_dir.exists():
        kwargs["text_recognition_model_dir"] = str(recognition_dir)
    return paddleocr_cls(**kwargs)


def ensure_paddleocr_model_env(config: OcrConfig) -> None:
    config.model_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(config.model_root))


def text_detection_model_dir(config: OcrConfig) -> Path:
    return config.model_root / "official_models" / DETECTION_MODEL_DIRNAME


def text_recognition_model_dir(config: OcrConfig) -> Path:
    return config.model_root / "official_models" / RECOGNITION_MODEL_DIRNAME


def parse_paddleocr_prediction(prediction: Any, confidence_threshold: float) -> list[dict[str, Any]]:
    texts = []
    for result in prediction or []:
        data = _result_to_dict(result)
        payload = data.get("res", data)
        rec_texts = _value_or_empty(payload.get("rec_texts"))
        rec_scores = _value_or_empty(payload.get("rec_scores"))
        rec_polys = _first_present(payload.get("rec_polys"), payload.get("dt_polys"))
        rec_boxes = _value_or_empty(payload.get("rec_boxes"))

        for index, text in enumerate(rec_texts):
            confidence = _safe_float(_sequence_item(rec_scores, index), default=0.0)
            if confidence < confidence_threshold:
                continue
            bbox = _to_plain_list(_sequence_item(rec_polys, index))
            if bbox is None:
                bbox = _to_plain_list(_sequence_item(rec_boxes, index))
            texts.append({"text": str(text), "confidence": confidence, "bbox": bbox})
    return texts


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


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "json"):
        data = result.json
        if callable(data):
            data = data()
        if isinstance(data, dict):
            return data
    if hasattr(result, "to_dict"):
        data = result.to_dict()
        if isinstance(data, dict):
            return data
    if hasattr(result, "__dict__"):
        data = vars(result)
        if isinstance(data, dict):
            return data
    return {}


def _sequence_item(value: Any, index: int) -> Any:
    if value is None:
        return None
    try:
        return value[index]
    except Exception:
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_plain_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_plain_list(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_list(item) for item in value]
    return value


def _value_or_empty(value: Any) -> Any:
    return [] if value is None else value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return []
