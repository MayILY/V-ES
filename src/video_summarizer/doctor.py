from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config, LlmConfig, OcrConfig, SummaryConfig, VisionConfig
from .llm import ProviderReadiness, check_provider_ready


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    suggestion: str


ModuleFinder = Callable[[str], object | None]
ExecutableFinder = Callable[[str], str | None]


def run_doctor(
    config: Config | None = None,
    ocr_config: OcrConfig | None = None,
    llm_config: LlmConfig | None = None,
    summary_config: SummaryConfig | None = None,
    vision_config: VisionConfig | None = None,
    module_finder: ModuleFinder | None = None,
    executable_finder: ExecutableFinder | None = None,
    env: dict[str, str] | None = None,
    path_exists: Callable[[Path], bool] | None = None,
) -> list[DoctorCheck]:
    module_finder = module_finder or importlib.util.find_spec
    executable_finder = executable_finder or shutil.which
    env = env if env is not None else os.environ
    path_exists = path_exists or Path.exists
    if config is not None:
        ocr_config = config.ocr
        llm_config = config.llm
        summary_config = config.summary
        vision_config = config.vision
    llm_config = llm_config or LlmConfig()
    summary_config = summary_config or SummaryConfig()
    vision_config = vision_config or VisionConfig()

    checks = [
        _executable_check("ffmpeg", "Required for audio extraction, frame sampling, and merge.", "Install ffmpeg and add it to PATH.", executable_finder),
        _executable_check("ffprobe", "Required for media metadata inspection.", "Install ffmpeg/ffprobe and add them to PATH.", executable_finder),
        _module_check("faster_whisper", "faster-whisper", "Required for transcript.json and transcript.srt.", 'python -m pip install -e ".[transcribe]"', module_finder),
        _module_check("paddle", "PaddlePaddle", "Required by PP-OCRv5 mobile local inference.", 'python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/', module_finder),
        _module_check("paddleocr", "PaddleOCR", "Required for real OCR output.", 'python -m pip install -e ".[ocr]"', module_finder),
        _module_check("scenedetect", "PySceneDetect", "Required for real scene detection with --scene-detect.", 'python -m pip install -e ".[scene]"', module_finder),
        _module_check("cv2", "OpenCV", "Required by PySceneDetect. This project pins opencv-contrib-python to match PP-OCRv5.", 'python -m pip install -e ".[scene]"', module_finder),
        _llm_check(
            "summary provider",
            check_provider_ready(summary_config.provider, summary_config.model, llm_config, env=env, module_finder=module_finder),
            "Required for final LLM summary.",
        ),
        _llm_check(
            "vision provider",
            check_provider_ready(
                vision_config.provider,
                vision_config.model,
                llm_config,
                need_vision=vision_config.enabled,
                env=env,
                module_finder=module_finder,
            ),
            "Required only when --vision or vision.enabled is used.",
            skipped=not vision_config.enabled,
        ),
        _module_check("PIL", "Pillow", "Required to resize images before vision calls.", 'python -m pip install -e ".[vision]"', module_finder),
    ]
    if ocr_config is not None:
        checks.append(_model_root_check(ocr_config.model_root, path_exists))
    return checks


def has_blocking_missing(checks: list[DoctorCheck]) -> bool:
    required = {"ffmpeg", "ffprobe", "faster-whisper"}
    return any(check.name in required and check.status == "missing" for check in checks)


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    lines = ["Dependency check:"]
    for check in checks:
        lines.append(f"- {check.status.upper():7} {check.name}: {check.detail}")
        if check.status != "ok" and check.suggestion:
            lines.append(f"  fix: {check.suggestion}")

    if has_blocking_missing(checks):
        lines.append("")
        lines.append("Result: missing required baseline dependencies.")
    elif any(check.status in {"missing", "warning"} for check in checks):
        lines.append("")
        lines.append("Result: baseline is usable, but full OCR/scene/LLM acceptance is incomplete.")
    else:
        lines.append("")
        lines.append("Result: all planned stable CLI dependencies are available.")
    return "\n".join(lines)


def _executable_check(
    name: str,
    ok_detail: str,
    suggestion: str,
    executable_finder: ExecutableFinder,
) -> DoctorCheck:
    path = executable_finder(name)
    if path:
        return DoctorCheck(name=name, status="ok", detail=f"{ok_detail} Found at {path}.", suggestion="")
    return DoctorCheck(name=name, status="missing", detail=ok_detail, suggestion=suggestion)


def _module_check(
    module_name: str,
    display_name: str,
    ok_detail: str,
    suggestion: str,
    module_finder: ModuleFinder,
) -> DoctorCheck:
    if module_finder(module_name):
        return DoctorCheck(name=display_name, status="ok", detail=ok_detail, suggestion="")
    return DoctorCheck(name=display_name, status="missing", detail=ok_detail, suggestion=suggestion)


def _env_check(
    name: str,
    ok_detail: str,
    suggestion: str,
    env: dict[str, str],
) -> DoctorCheck:
    if env.get(name):
        return DoctorCheck(name=name, status="ok", detail=ok_detail, suggestion="")
    return DoctorCheck(name=name, status="missing", detail=ok_detail, suggestion=suggestion)


def _llm_check(name: str, readiness: ProviderReadiness, ok_detail: str, skipped: bool = False) -> DoctorCheck:
    display_name = f"{name} ({readiness.provider})"
    if skipped:
        return DoctorCheck(
            name=display_name,
            status="skipped",
            detail=f"{ok_detail} Vision is disabled.",
            suggestion="",
        )
    if readiness.status == "ok":
        return DoctorCheck(name=display_name, status="ok", detail=ok_detail, suggestion="")
    if readiness.status == "warning":
        return DoctorCheck(
            name=display_name,
            status="warning",
            detail=f"{ok_detail} {readiness.reason}.",
            suggestion=readiness.suggestion,
        )
    return DoctorCheck(
        name=display_name,
        status="missing",
        detail=f"{ok_detail} {readiness.reason}.",
        suggestion=readiness.suggestion,
    )


def _model_root_check(model_root: Path, path_exists: Callable[[Path], bool]) -> DoctorCheck:
    det_dir = model_root / "official_models" / "PP-OCRv5_mobile_det"
    rec_dir = model_root / "official_models" / "PP-OCRv5_mobile_rec"
    if path_exists(det_dir) and path_exists(rec_dir):
        return DoctorCheck(
            name="PP-OCRv5 model_root",
            status="ok",
            detail=f"Local PP-OCRv5 mobile models exist at {model_root}.",
            suggestion="",
        )
    return DoctorCheck(
        name="PP-OCRv5 model_root",
        status="missing",
        detail=f"Local model cache is not prepared at {model_root}.",
        suggestion="Run video-summary ocr-prepare.",
    )
