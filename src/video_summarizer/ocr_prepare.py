from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import OcrConfig
from .io_utils import write_json
from .ocr import create_paddleocr_engine, ensure_paddleocr_model_env, parse_paddleocr_prediction


def prepare_ocr(config: OcrConfig) -> dict[str, Any]:
    ensure_paddleocr_model_env(config)

    try:
        import paddle  # type: ignore
    except Exception as exc:
        return _failed("paddle_unavailable", exc, config)

    try:
        paddle.utils.run_check()
    except Exception as exc:
        return _failed("paddle_check_failed", exc, config)

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        return _failed("paddleocr_unavailable", exc, config)

    try:
        smoke_image = _create_smoke_image(config.model_root / "_smoke" / "ppocrv5_smoke.png")
        engine = create_paddleocr_engine(PaddleOCR, config)
        prediction = engine.predict(str(smoke_image)) or []
        texts = parse_paddleocr_prediction(prediction, 0.0)
        status = {
            "status": "ok",
            "engine": config.engine,
            "version": config.version,
            "device": config.device,
            "model_root": str(config.model_root),
            "text_detection_model_name": config.text_detection_model_name,
            "text_recognition_model_name": config.text_recognition_model_name,
            "smoke_image": str(smoke_image),
            "recognized_texts": texts,
        }
        write_json(config.model_root / "ocr_prepare_status.json", status)
        return status
    except Exception as exc:
        return _failed("ocr_prepare_failed", exc, config)


def _create_smoke_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow is required for OCR smoke image generation: {exc}") from exc

    image = Image.new("RGB", (640, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((32, 56), "PP-OCRv5 local test 123", fill="black")
    image.save(path)
    return path


def _failed(reason: str, exc: Exception, config: OcrConfig) -> dict[str, Any]:
    status = {
        "status": "failed",
        "reason": reason,
        "error": str(exc),
        "model_root": str(config.model_root),
        "text_detection_model_name": config.text_detection_model_name,
        "text_recognition_model_name": config.text_recognition_model_name,
    }
    config.model_root.mkdir(parents=True, exist_ok=True)
    write_json(config.model_root / "ocr_prepare_status.json", status)
    return status
