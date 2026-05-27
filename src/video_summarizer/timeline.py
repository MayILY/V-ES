from __future__ import annotations

from typing import Any


def build_timeline(
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    ocr: dict[str, Any],
    frames: list[dict[str, Any]],
    window_sec: float,
) -> dict[str, Any]:
    duration = metadata.get("duration_sec") or _max_end(transcript, frames, ocr)
    if not duration or duration <= 0:
        duration = window_sec
    events = []
    start = 0.0
    while start < duration:
        end = min(start + window_sec, duration)
        segments = _segments_in_window(transcript.get("segments", []), start, end)
        ocr_frames = _ocr_in_window(ocr.get("frames", []), start, end)
        frame_items = _frames_in_window(frames, start, end)
        events.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "transcript": " ".join(s.get("text", "") for s in segments).strip(),
                "transcript_segments": segments,
                "ocr_text": " / ".join(
                    f.get("raw_text", "") for f in ocr_frames if f.get("raw_text") and not f.get("is_duplicate")
                ),
                "ocr_frames": ocr_frames,
                "frames": frame_items,
                "event_summary": "",
            }
        )
        start = end
        if start == duration:
            break
    return {"window_sec": window_sec, "events": events}


def _segments_in_window(segments: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [s for s in segments if float(s.get("end", 0)) > start and float(s.get("start", 0)) < end]


def _ocr_in_window(frames: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [f for f in frames if start <= float(f.get("timestamp", 0)) < end]


def _frames_in_window(frames: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [f for f in frames if start <= float(f.get("timestamp", 0)) < end]


def _max_end(transcript: dict[str, Any], frames: list[dict[str, Any]], ocr: dict[str, Any]) -> float:
    values = [0.0]
    values.extend(float(s.get("end", 0)) for s in transcript.get("segments", []))
    values.extend(float(f.get("timestamp", 0)) for f in frames)
    values.extend(float(f.get("timestamp", 0)) for f in ocr.get("frames", []))
    return max(values)
