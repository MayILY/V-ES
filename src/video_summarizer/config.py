from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class OutputConfig:
    base_dir: Path = Path("outputs")
    overwrite: bool = False
    keep_intermediate_files: bool = True


@dataclass
class FfmpegConfig:
    frame_interval_sec: float = 5.0
    audio_sample_rate: int = 16000
    max_frame_width: int = 1280


@dataclass
class WhisperConfig:
    model: str = "medium"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "zh"
    vad_filter: bool = True


@dataclass
class OcrConfig:
    engine: str = "paddleocr"
    language: str = "ch"
    confidence_threshold: float = 0.5
    deduplicate: bool = True
    duplicate_similarity_threshold: float = 0.9


@dataclass
class SummaryConfig:
    model: str = "gpt-4.1-mini"
    segment_window_sec: float = 60.0
    output_language: str = "zh"
    include_uncertainties: bool = True


@dataclass
class SceneDetectionConfig:
    enabled: bool = False
    threshold: float = 27.0
    min_scene_len_sec: float = 2.0
    max_keyframes_per_scene: int = 3
    duplicate_similarity_threshold: float = 0.98


@dataclass
class VisionConfig:
    enabled: bool = False
    model: str = "gpt-4.1-mini"
    max_frames: int = 80
    max_image_width: int = 1280
    detail: str = "low"


@dataclass
class Config:
    output: OutputConfig = field(default_factory=OutputConfig)
    ffmpeg: FfmpegConfig = field(default_factory=FfmpegConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


def load_config(path: Optional[Path] = None) -> Config:
    cfg = Config()
    path = path or Path("config.yaml")
    if not path.exists():
        return cfg

    try:
        import yaml  # type: ignore
    except Exception:
        return cfg

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _merge_dataclass(cfg.output, raw.get("output", {}))
    _merge_dataclass(cfg.ffmpeg, raw.get("ffmpeg", {}))
    _merge_dataclass(cfg.whisper, raw.get("whisper", {}))
    _merge_dataclass(cfg.ocr, raw.get("ocr", {}))
    _merge_dataclass(cfg.summary, raw.get("summary", {}))
    _merge_dataclass(cfg.scene_detection, raw.get("scene_detection", {}))
    _merge_dataclass(cfg.vision, raw.get("vision", {}))
    if isinstance(cfg.output.base_dir, str):
        cfg.output.base_dir = Path(cfg.output.base_dir)
    return cfg


def _merge_dataclass(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)
