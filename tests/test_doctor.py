from __future__ import annotations

from typer.testing import CliRunner

from video_summarizer import main
from video_summarizer.config import Config, OcrConfig
from video_summarizer.doctor import DoctorCheck, format_doctor_report, has_blocking_missing, run_doctor


def test_doctor_reports_all_ok_when_dependencies_are_present() -> None:
    config = Config()
    config.vision.enabled = True
    checks = run_doctor(
        config=config,
        module_finder=lambda name: object(),
        executable_finder=lambda name: f"C:/tools/{name}.exe",
        env={"OPENAI_API_KEY": "test-key"},
    )

    assert all(check.status == "ok" for check in checks)
    assert not has_blocking_missing(checks)
    assert "all planned stable CLI dependencies are available" in format_doctor_report(checks)


def test_doctor_distinguishes_required_baseline_from_optional_acceptance() -> None:
    missing_modules = {"paddle", "paddleocr", "scenedetect", "cv2", "openai"}

    checks = run_doctor(
        module_finder=lambda name: None if name in missing_modules else object(),
        executable_finder=lambda name: f"C:/tools/{name}.exe",
        env={},
    )

    by_name = {check.name: check for check in checks}
    assert by_name["ffmpeg"].status == "ok"
    assert by_name["ffprobe"].status == "ok"
    assert by_name["faster-whisper"].status == "ok"
    assert by_name["PaddlePaddle"].status == "missing"
    assert by_name["PaddleOCR"].status == "missing"
    assert by_name["PySceneDetect"].status == "missing"
    assert by_name["OpenCV"].status == "missing"
    assert by_name["summary provider (openai)"].status == "missing"
    assert by_name["vision provider (openai)"].status == "skipped"
    assert not has_blocking_missing(checks)
    assert "full OCR/scene/LLM acceptance is incomplete" in format_doctor_report(checks)


def test_doctor_warns_when_enabled_vision_provider_has_no_vision_support() -> None:
    config = Config()
    config.vision.enabled = True
    config.vision.provider = "deepseek"
    config.vision.model = "deepseek-v4-pro"

    checks = run_doctor(
        config=config,
        module_finder=lambda name: object(),
        executable_finder=lambda name: f"C:/tools/{name}.exe",
        env={"OPENAI_API_KEY": "test-key", "DEEPSEEK_API_KEY": "test-key"},
    )

    by_name = {check.name: check for check in checks}
    assert by_name["vision provider (deepseek)"].status == "warning"
    assert "provider_vision_unsupported" in by_name["vision provider (deepseek)"].detail


def test_doctor_marks_missing_baseline_dependency_as_blocking() -> None:
    checks = run_doctor(
        module_finder=lambda name: None if name == "faster_whisper" else object(),
        executable_finder=lambda name: None if name == "ffmpeg" else f"C:/tools/{name}.exe",
        env={"OPENAI_API_KEY": "test-key"},
    )

    assert has_blocking_missing(checks)
    report = format_doctor_report(checks)
    assert "MISSING ffmpeg" in report
    assert "MISSING faster-whisper" in report
    assert "missing required baseline dependencies" in report


def test_doctor_cli_prints_report_and_uses_blocking_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "run_doctor",
        lambda **kwargs: [
            DoctorCheck(
                name="ffmpeg",
                status="missing",
                detail="Required for audio extraction.",
                suggestion="Install ffmpeg.",
            )
        ],
    )

    result = CliRunner().invoke(main.app, ["doctor"])

    assert result.exit_code == 1
    assert "Dependency check:" in result.output
    assert "MISSING ffmpeg" in result.output


def test_run_cli_rejects_conflicting_cache_modes(tmp_path) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"")

    result = CliRunner().invoke(main.app, ["run", str(media), "--no-llm-cache", "--refresh-llm-cache"])

    assert result.exit_code != 0
    assert "mutually" in result.output
    assert "exclusive" in result.output


def test_doctor_reports_missing_ppocrv5_model_root() -> None:
    checks = run_doctor(
        ocr_config=OcrConfig(),
        module_finder=lambda name: object(),
        executable_finder=lambda name: f"C:/tools/{name}.exe",
        env={"OPENAI_API_KEY": "test-key"},
        path_exists=lambda path: False,
    )

    by_name = {check.name: check for check in checks}
    assert by_name["PP-OCRv5 model_root"].status == "missing"
    assert by_name["PP-OCRv5 model_root"].suggestion == "Run video-summary ocr-prepare."
