from __future__ import annotations

import os
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
    version: str = "PP-OCRv5"
    language: str = "ch"
    device: str = "cpu"
    model_root: Path = Path("D:/someElse/video_summarizer-models/paddleocr")
    text_detection_model_name: str = "PP-OCRv5_mobile_det"
    text_recognition_model_name: str = "PP-OCRv5_mobile_rec"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    confidence_threshold: float = 0.5
    deduplicate: bool = True
    duplicate_similarity_threshold: float = 0.9


@dataclass
class SummaryConfig:
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    segment_window_sec: float = 60.0
    chapter_window_sec: float = 300.0
    max_events_per_request: int = 40
    output_language: str = "zh"
    summary_style: str = "structured"
    include_uncertainties: bool = True


@dataclass
class EvidenceConfig:
    enabled: bool = True
    min_text_info_chars: int = 12
    preserve_chapter_boundaries: bool = True
    preserve_numeric_differences: bool = True
    max_chapter_summary_chars: int = 1200
    max_prompt_chars: int = 12000
    evidence_schema_version: str = "summary-evidence-v1"
    builder_version: str = "evidence-builder-v1"
    prompt_template_version: str = "summary-prompt-v1"
    event_duplicate_similarity_threshold: float = 0.92
    keyword_signal_terms: list[str] = field(default_factory=lambda: ["error", "warning", "失败", "成功", "金额", "版本", "设置", "登录", "支付"])


@dataclass
class SceneDetectionConfig:
    enabled: bool = False
    detector: str = "content"
    threshold: float = 27.0
    min_scene_len_sec: float = 2.0
    adaptive_threshold: float = 3.0
    min_content_val: float = 15.0
    window_width: int = 2
    max_keyframes_per_scene: int = 3
    duplicate_similarity_threshold: float = 0.98


@dataclass
class VisionConfig:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    max_frames: int = 80
    max_image_width: int = 1280
    detail: str = "low"
    prompt_template_version: str = "vision-prompt-v1"
    image_preprocessing_version: str = "image-preprocess-v1"


@dataclass
class LlmProviderConfig:
    type: str = "openai_compatible"
    base_url: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    api_key_required: bool = True
    supports_text: bool = True
    supports_vision: bool = False
    vision_format: str = "none"
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmConfig:
    env_file: Path | None = Path(".env")
    override_env: bool = False
    providers: dict[str, LlmProviderConfig] = field(default_factory=lambda: default_llm_providers())


@dataclass
class LlmCacheConfig:
    mode: str = "read_write"
    dir: Path = Path("outputs/_cache/llm")
    schema_version: str = "llm-cache-v1"


@dataclass
class Config:
    output: OutputConfig = field(default_factory=OutputConfig)
    ffmpeg: FfmpegConfig = field(default_factory=FfmpegConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    llm_cache: LlmCacheConfig = field(default_factory=LlmCacheConfig)


def default_llm_providers() -> dict[str, LlmProviderConfig]:
    return {
        "openai": LlmProviderConfig(
            type="openai_compatible",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            api_key_required=True,
            supports_text=True,
            supports_vision=True,
            vision_format="responses",
        ),
        "deepseek": LlmProviderConfig(
            type="openai_compatible",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            api_key_required=True,
            supports_text=True,
            supports_vision=False,
            vision_format="none",
            reasoning_effort="max",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        "qwen": LlmProviderConfig(
            type="openai_compatible",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            api_key_required=True,
            supports_text=True,
            supports_vision=False,
            vision_format="none",
        ),
        "gemini": LlmProviderConfig(
            type="gemini",
            base_url=None,
            api_key_env="GEMINI_API_KEY",
            api_key_required=True,
            supports_text=True,
            supports_vision=False,
            vision_format="none",
        ),
        "local": LlmProviderConfig(
            type="openai_compatible",
            base_url="http://localhost:11434/v1",
            api_key_env=None,
            api_key_required=False,
            supports_text=True,
            supports_vision=False,
            vision_format="none",
        ),
    }


def load_config(path: Optional[Path] = None) -> Config:
    cfg = Config()
    path = path or Path("config.yaml")
    if not path.exists():
        load_env_file(cfg.llm.env_file, cfg.llm.override_env)
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
    _merge_dataclass(cfg.evidence, raw.get("evidence", {}))
    _merge_dataclass(cfg.scene_detection, raw.get("scene_detection", {}))
    _merge_dataclass(cfg.vision, raw.get("vision", {}))
    _merge_llm_config(cfg.llm, raw.get("llm", {}))
    _merge_dataclass(cfg.llm_cache, raw.get("llm_cache", {}))
    if isinstance(cfg.output.base_dir, str):
        cfg.output.base_dir = Path(cfg.output.base_dir)
    if isinstance(cfg.ocr.model_root, str):
        cfg.ocr.model_root = Path(cfg.ocr.model_root)
    if isinstance(cfg.llm_cache.dir, str):
        cfg.llm_cache.dir = Path(cfg.llm_cache.dir)
    if isinstance(cfg.llm.env_file, str):
        cfg.llm.env_file = Path(cfg.llm.env_file) if cfg.llm.env_file else None
    if cfg.llm.env_file is not None and not cfg.llm.env_file.is_absolute():
        cfg.llm.env_file = path.parent / cfg.llm.env_file
    load_env_file(cfg.llm.env_file, cfg.llm.override_env)
    return cfg


def _merge_dataclass(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)


def _merge_llm_config(target: LlmConfig, values: dict[str, Any]) -> None:
    if isinstance(values, dict):
        if "env_file" in values:
            target.env_file = values["env_file"]
        if "override_env" in values:
            target.override_env = bool(values["override_env"])
    providers = values.get("providers", {}) if isinstance(values, dict) else {}
    for name, raw_provider in providers.items():
        if not isinstance(raw_provider, dict):
            continue
        provider = target.providers.get(name, LlmProviderConfig())
        _merge_dataclass(provider, raw_provider)
        target.providers[name] = provider


def load_env_file(
    path: Path | str | None,
    override: bool = False,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    if path is None:
        return {}
    env = environ if environ is not None else os.environ
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in env:
            env[key] = value
            loaded[key] = value
    return loaded


def _parse_env_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value
