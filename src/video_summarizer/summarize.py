from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import SummaryConfig


def summarize_timeline(timeline: dict[str, Any], output_path: Path, config: SummaryConfig, skip: bool = False) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skip:
        text = _fallback_markdown(timeline, "summary_skipped")
        output_path.write_text(text, encoding="utf-8")
        return {"status": "skipped", "reason": "requested", "path": str(output_path)}

    if not os.environ.get("OPENAI_API_KEY"):
        text = _fallback_markdown(timeline, "missing_openai_api_key")
        output_path.write_text(text, encoding="utf-8")
        return {"status": "skipped", "reason": "missing_openai_api_key", "path": str(output_path)}

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        text = _fallback_markdown(timeline, "openai_package_unavailable")
        output_path.write_text(text, encoding="utf-8")
        return {"status": "skipped", "reason": f"openai_package_unavailable: {exc}", "path": str(output_path)}

    prompt = _build_prompt(timeline, config)
    try:
        client = OpenAI()
        response = client.responses.create(model=config.model, input=prompt)
        text = getattr(response, "output_text", None) or _extract_response_text(response)
        output_path.write_text(text.strip() + "\n", encoding="utf-8")
        return {"status": "ok", "path": str(output_path), "model": config.model}
    except Exception as exc:
        text = _fallback_markdown(timeline, f"openai_call_failed: {exc}")
        output_path.write_text(text, encoding="utf-8")
        return {"status": "failed", "reason": str(exc), "path": str(output_path)}


def _build_prompt(timeline: dict[str, Any], config: SummaryConfig) -> str:
    events = []
    for event in timeline.get("events", []):
        events.append(
            {
                "start": event.get("start"),
                "end": event.get("end"),
                "transcript": event.get("transcript"),
                "ocr_text": event.get("ocr_text"),
            }
        )
    return (
        "你是一个视频内容总结助手。只基于下面的结构化时间线证据生成 Markdown 总结，"
        "不要编造未出现的信息。请包含：一句话概括、核心内容、时间线、重要文字、可能遗漏或不确定的信息。\n\n"
        f"输出语言：{config.output_language}\n"
        f"时间线证据：{events}"
    )


def _fallback_markdown(timeline: dict[str, Any], reason: str) -> str:
    lines = [
        "# 视频总结",
        "",
        f"> 自动总结未执行：{reason}",
        "",
        "## 时间线证据",
        "",
        "| 时间 | 转录 | 画面文字 |",
        "|---|---|---|",
    ]
    for event in timeline.get("events", []):
        time_range = f"{event.get('start')} - {event.get('end')}s"
        transcript = (event.get("transcript") or "").replace("|", "\\|")
        ocr_text = (event.get("ocr_text") or "").replace("|", "\\|")
        lines.append(f"| {time_range} | {transcript} | {ocr_text} |")
    lines.append("")
    return "\n".join(lines)


def _extract_response_text(response: Any) -> str:
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)
