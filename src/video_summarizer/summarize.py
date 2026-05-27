from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import SummaryConfig


def summarize_timeline(
    timeline: dict[str, Any],
    output_path: Path,
    config: SummaryConfig,
    skip: bool = False,
    timeline_summary_path: Path | None = None,
    chapter_summaries_path: Path | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_summary_path = timeline_summary_path or output_path.with_name("timeline_summary.md")
    chapter_summaries_path = chapter_summaries_path or output_path.with_name("chapter_summaries.md")

    chapters = build_chapters(timeline, config.chapter_window_sec)
    timeline_summary_path.write_text(_timeline_markdown(timeline), encoding="utf-8")
    chapter_summaries_path.write_text(_chapter_markdown(chapters), encoding="utf-8")

    common_status = {
        "timeline_summary": {"status": "ok", "path": str(timeline_summary_path)},
        "chapter_summaries": {"status": "ok", "path": str(chapter_summaries_path), "chapter_count": len(chapters)},
    }

    if skip:
        text = _fallback_markdown(timeline, chapters, "summary_skipped")
        output_path.write_text(text, encoding="utf-8")
        return {"status": "skipped", "reason": "requested", "path": str(output_path), **common_status}

    if not os.environ.get("OPENAI_API_KEY"):
        text = _fallback_markdown(timeline, chapters, "missing_openai_api_key")
        output_path.write_text(text, encoding="utf-8")
        return {"status": "skipped", "reason": "missing_openai_api_key", "path": str(output_path), **common_status}

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        text = _fallback_markdown(timeline, chapters, "openai_package_unavailable")
        output_path.write_text(text, encoding="utf-8")
        return {
            "status": "skipped",
            "reason": f"openai_package_unavailable: {exc}",
            "path": str(output_path),
            **common_status,
        }

    prompt = _build_prompt(timeline, chapters, config)
    try:
        client = OpenAI()
        response = client.responses.create(model=config.model, input=prompt)
        text = getattr(response, "output_text", None) or _extract_response_text(response)
        output_path.write_text(text.strip() + "\n", encoding="utf-8")
        return {"status": "ok", "path": str(output_path), "model": config.model, **common_status}
    except Exception as exc:
        text = _fallback_markdown(timeline, chapters, f"openai_call_failed: {exc}")
        output_path.write_text(text, encoding="utf-8")
        return {"status": "failed", "reason": str(exc), "path": str(output_path), **common_status}


def build_chapters(timeline: dict[str, Any], chapter_window_sec: float) -> list[dict[str, Any]]:
    events = timeline.get("events", [])
    if not events:
        return []

    chapter_window_sec = chapter_window_sec if chapter_window_sec > 0 else timeline.get("window_sec", 60.0)
    chapters: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for event in events:
        start = float(event.get("start", 0.0))
        end = float(event.get("end", start))
        if current is None or start >= float(current["start"]) + chapter_window_sec:
            if current is not None:
                _finalize_chapter(current)
                chapters.append(current)
            current = {
                "chapter_id": len(chapters) + 1,
                "start": start,
                "end": end,
                "events": [],
                "transcript": [],
                "ocr_text": [],
                "visual_text": [],
            }
        current["end"] = max(float(current["end"]), end)
        current["events"].append({"start": event.get("start"), "end": event.get("end")})
        _append_if_present(current["transcript"], event.get("transcript"))
        _append_if_present(current["ocr_text"], event.get("ocr_text"))
        _append_if_present(current["visual_text"], event.get("visual_text"))

    if current is not None:
        _finalize_chapter(current)
        chapters.append(current)
    return chapters


def _finalize_chapter(chapter: dict[str, Any]) -> None:
    chapter["start"] = round(float(chapter["start"]), 3)
    chapter["end"] = round(float(chapter["end"]), 3)
    chapter["transcript"] = " ".join(chapter["transcript"]).strip()
    chapter["ocr_text"] = " / ".join(chapter["ocr_text"]).strip()
    chapter["visual_text"] = " / ".join(chapter["visual_text"]).strip()


def _append_if_present(target: list[str], value: Any) -> None:
    if value:
        target.append(str(value).strip())


def _build_prompt(timeline: dict[str, Any], chapters: list[dict[str, Any]], config: SummaryConfig) -> str:
    events = []
    for event in timeline.get("events", [])[: config.max_events_per_request]:
        events.append(_event_evidence(event))

    return (
        "你是一个视频内容总结助手。只基于下面的结构化证据生成 Markdown 总结，"
        "不要编造未出现的信息。请包含：一句话概括、核心内容、章节摘要、时间线、"
        "重要文字、视觉证据、可能遗漏或不确定的信息。\n\n"
        f"输出语言：{config.output_language}\n"
        f"章节证据：{[_chapter_evidence(chapter) for chapter in chapters]}\n"
        f"时间线证据：{events}"
    )


def _timeline_markdown(timeline: dict[str, Any]) -> str:
    lines = [
        "# 时间线摘要",
        "",
        "| 时间 | 转录 | 画面文字 | 视觉描述 |",
        "|---|---|---|---|",
    ]
    for event in timeline.get("events", []):
        lines.append(
            "| {time} | {transcript} | {ocr} | {visual} |".format(
                time=f"{event.get('start')} - {event.get('end')}s",
                transcript=_cell(event.get("transcript")),
                ocr=_cell(event.get("ocr_text")),
                visual=_cell(event.get("visual_text")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _chapter_markdown(chapters: list[dict[str, Any]]) -> str:
    lines = ["# 章节摘要", ""]
    if not chapters:
        lines.extend(["> 没有可用章节证据。", ""])
        return "\n".join(lines)

    for chapter in chapters:
        lines.extend(
            [
                f"## Chapter {chapter['chapter_id']}: {chapter['start']} - {chapter['end']}s",
                "",
                f"- 转录：{chapter.get('transcript') or '无'}",
                f"- 画面文字：{chapter.get('ocr_text') or '无'}",
                f"- 视觉描述：{chapter.get('visual_text') or '无'}",
                "",
            ]
        )
    return "\n".join(lines)


def _fallback_markdown(timeline: dict[str, Any], chapters: list[dict[str, Any]], reason: str) -> str:
    lines = [
        "# 视频总结",
        "",
        f"> 自动最终总结未执行：{reason}",
        "",
        "## 章节证据",
        "",
    ]
    if chapters:
        for chapter in chapters:
            lines.append(
                f"- {chapter['start']} - {chapter['end']}s: "
                f"转录={chapter.get('transcript') or '无'}；"
                f"画面文字={chapter.get('ocr_text') or '无'}；"
                f"视觉描述={chapter.get('visual_text') or '无'}"
            )
    else:
        lines.append("- 无可用章节证据")
    lines.extend(["", "## 时间线证据", "", _timeline_markdown(timeline)])
    return "\n".join(lines)


def _event_evidence(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": event.get("start"),
        "end": event.get("end"),
        "transcript": event.get("transcript"),
        "ocr_text": event.get("ocr_text"),
        "visual_text": event.get("visual_text"),
    }


def _chapter_evidence(chapter: dict[str, Any]) -> dict[str, Any]:
    return {
        "chapter_id": chapter.get("chapter_id"),
        "start": chapter.get("start"),
        "end": chapter.get("end"),
        "transcript": chapter.get("transcript"),
        "ocr_text": chapter.get("ocr_text"),
        "visual_text": chapter.get("visual_text"),
    }


def _cell(value: Any) -> str:
    return (str(value or "")).replace("|", "\\|").replace("\n", " ")


def _extract_response_text(response: Any) -> str:
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)
