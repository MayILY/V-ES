from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import WhisperConfig


def transcribe_audio(audio_path: Path, config: WhisperConfig) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:
        return empty_transcript("faster_whisper_unavailable", str(exc), language=config.language)

    try:
        model = WhisperModel(config.model, device=config.device, compute_type=config.compute_type)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=config.language or None,
            vad_filter=config.vad_filter,
        )
        segments = [
            {"id": idx, "start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
            for idx, segment in enumerate(segments_iter, start=1)
        ]
        return {
            "status": "ok",
            "language": getattr(info, "language", config.language),
            "duration": getattr(info, "duration", None),
            "segments": segments,
        }
    except Exception as exc:
        return empty_transcript("transcription_failed", str(exc), language=config.language)


def empty_transcript(status: str, reason: str, language: str = "zh") -> dict[str, Any]:
    return {"status": status, "reason": reason, "language": language, "segments": []}


def write_srt(path: Path, transcript: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, segment in enumerate(transcript.get("segments", []), start=1):
        lines.extend(
            [
                str(index),
                f"{format_srt_time(segment['start'])} --> {format_srt_time(segment['end'])}",
                segment.get("text", ""),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
