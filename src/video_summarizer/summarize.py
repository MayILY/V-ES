from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import EvidenceConfig, LlmCacheConfig, LlmConfig, SummaryConfig
from .evidence import SummaryEvidence, build_summary_evidence, evidence_report_markdown
from .io_utils import write_json
from .llm import LlmClient, LlmProviderError, check_provider_ready, create_llm_client
from .llm_cache import cache_key, hash_text, lookup_text_cache, provider_cache_params, write_text_cache

FORBIDDEN_PROMPT_TOKENS = ("frames", "ocr_frames", "transcript_segments", "image_path", "frame_")


def summarize_timeline(
    timeline: dict[str, Any],
    output_path: Path,
    config: SummaryConfig,
    llm_config: LlmConfig | None = None,
    evidence_config: EvidenceConfig | None = None,
    llm_cache_config: LlmCacheConfig | None = None,
    skip: bool = False,
    timeline_summary_path: Path | None = None,
    chapter_summaries_path: Path | None = None,
    summary_evidence_json_path: Path | None = None,
    summary_evidence_md_path: Path | None = None,
    llm_client: LlmClient | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_summary_path = timeline_summary_path or output_path.with_name("timeline_summary.md")
    chapter_summaries_path = chapter_summaries_path or output_path.with_name("chapter_summaries.md")
    evidence_config = evidence_config or EvidenceConfig()
    llm_config = llm_config or LlmConfig()
    llm_cache_config = llm_cache_config or LlmCacheConfig(mode="off")

    chapters = build_chapters(timeline, config.chapter_window_sec)
    evidence = build_summary_evidence(timeline, evidence_config, config)
    evidence.config["llm_cache_mode"] = llm_cache_config.mode

    timeline_summary_path.write_text(_timeline_markdown(timeline), encoding="utf-8")
    chapter_summaries_path.write_text(_chapter_markdown(chapters), encoding="utf-8")
    if summary_evidence_json_path is not None:
        write_json(summary_evidence_json_path, evidence.to_dict())
    if summary_evidence_md_path is not None:
        summary_evidence_md_path.parent.mkdir(parents=True, exist_ok=True)
        summary_evidence_md_path.write_text(evidence_report_markdown(evidence), encoding="utf-8")

    common_status = {
        "enabled": not skip,
        "called": False,
        "call_count": 0,
        "cache_mode": llm_cache_config.mode,
        "cache_hit": False,
        "cache_miss": False,
        "cache_write": False,
        "path": str(output_path),
        "timeline_summary": {"status": "ok", "path": str(timeline_summary_path)},
        "chapter_summaries": {"status": "ok", "path": str(chapter_summaries_path), "chapter_count": len(chapters)},
        "evidence": _evidence_status(evidence),
    }
    provider_status = {"provider": config.provider, "model": config.model}

    if skip:
        output_path.write_text(_fallback_markdown(evidence, "summary_skipped"), encoding="utf-8")
        return {"status": "skipped", "reason": "requested", **provider_status, **common_status}

    prompt = _build_prompt(evidence, config)
    provider_config = llm_config.providers.get(config.provider.lower().strip())
    key_payload = {
        "cache_schema_version": llm_cache_config.schema_version,
        "provider": config.provider,
        "model": config.model,
        "prompt_template_version": evidence.prompt_template_version,
        "evidence_schema_version": evidence.schema_version,
        "evidence_builder_version": evidence.builder_version,
        "output_language": config.output_language,
        "summary_style": config.summary_style,
        "provider_params": provider_cache_params(provider_config),
        "prompt_hash": hash_text(prompt),
    }
    key = cache_key(key_payload)
    lookup = lookup_text_cache(llm_cache_config, "summary", key)
    if lookup.hit:
        output_path.write_text((lookup.text or "").strip() + "\n", encoding="utf-8")
        return {
            "status": "ok",
            **provider_status,
            **common_status,
            "cache_hit": True,
            "cache_key": key,
            "cache_path": str(lookup.path),
        }

    readiness = check_provider_ready(config.provider, config.model, llm_config)
    if readiness.status != "ok":
        reason = readiness.reason or "provider_unavailable"
        output_path.write_text(_fallback_markdown(evidence, reason), encoding="utf-8")
        return {
            "status": "failed",
            "reason": reason,
            **provider_status,
            **common_status,
            "cache_miss": llm_cache_config.mode != "off",
            "cache_key": key,
        }

    try:
        client = llm_client or create_llm_client(config.provider, config.model, llm_config)
        text = client.generate_text(prompt)
        output_path.write_text(text.strip() + "\n", encoding="utf-8")
        cache_path = write_text_cache(llm_cache_config, "summary", key, text.strip(), key_payload)
        return {
            "status": "ok",
            **provider_status,
            **common_status,
            "called": True,
            "call_count": 1,
            "cache_miss": llm_cache_config.mode != "off",
            "cache_write": cache_path is not None,
            "cache_key": key,
            "cache_path": str(cache_path) if cache_path else None,
        }
    except (LlmProviderError, Exception) as exc:
        output_path.write_text(_fallback_markdown(evidence, f"llm_call_failed: {exc}"), encoding="utf-8")
        return {
            "status": "failed",
            "reason": str(exc),
            **provider_status,
            **common_status,
            "called": True,
            "call_count": 1,
            "cache_miss": llm_cache_config.mode != "off",
            "cache_key": key,
        }


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


def _build_prompt(evidence: SummaryEvidence, config: SummaryConfig) -> str:
    if not isinstance(evidence, SummaryEvidence):
        raise TypeError("_build_prompt expects SummaryEvidence")
    payload = {
        "schema_version": evidence.schema_version,
        "builder_version": evidence.builder_version,
        "chapters": [
            {
                "chapter_id": chapter.chapter_id,
                "start": chapter.start,
                "end": chapter.end,
                "transcript_summary": chapter.transcript_summary,
                "ocr_summary": chapter.ocr_summary,
                "visual_summary": chapter.visual_summary,
                "events": [
                    {
                        "start": event.start,
                        "end": event.end,
                        "transcript": event.transcript,
                        "ocr_text": event.ocr_text,
                        "visual_text": event.visual_text,
                        "score": event.score,
                        "keep_reason": event.keep_reason,
                    }
                    for event in chapter.selected_events
                ],
            }
            for chapter in evidence.chapters
        ],
        "stats": evidence.stats,
    }
    prompt = (
        "You are a video content summarization assistant. Generate a Markdown summary only from the structured evidence below. "
        "Do not invent facts that are not present. Include: one-sentence overview, core content, chapter summaries, timeline, "
        "important text, visual evidence, and uncertainties.\n\n"
        f"Output language: {config.output_language}\n"
        f"Summary style: {config.summary_style}\n"
        f"Evidence: {payload}"
    )
    _assert_prompt_safe(prompt)
    return prompt


def _assert_prompt_safe(prompt: str) -> None:
    lowered = prompt.lower()
    forbidden = [token for token in FORBIDDEN_PROMPT_TOKENS if token in lowered]
    assert not forbidden, f"forbidden raw evidence fields in prompt: {forbidden}"


def _evidence_status(evidence: SummaryEvidence) -> dict[str, Any]:
    stats = evidence.stats
    return {
        "schema_version": evidence.schema_version,
        "builder_version": evidence.builder_version,
        "prompt_template_version": evidence.prompt_template_version,
        "input_event_count": stats.get("input_event_count", 0),
        "selected_event_count": stats.get("selected_event_count", 0),
        "deduplicated_event_count": stats.get("deduplicated_event_count", 0),
        "low_info_filtered_count": stats.get("low_info_filtered_count", 0),
        "global_trimmed_event_count": stats.get("global_trimmed_event_count", 0),
        "prompt_chars": stats.get("estimated_prompt_chars", 0),
        "global_trimmed": stats.get("global_trimmed", False),
        "trim_reasons": stats.get("trim_reasons", []),
    }


def _finalize_chapter(chapter: dict[str, Any]) -> None:
    chapter["start"] = round(float(chapter["start"]), 3)
    chapter["end"] = round(float(chapter["end"]), 3)
    chapter["transcript"] = " ".join(chapter["transcript"]).strip()
    chapter["ocr_text"] = " / ".join(chapter["ocr_text"]).strip()
    chapter["visual_text"] = " / ".join(chapter["visual_text"]).strip()


def _append_if_present(target: list[str], value: Any) -> None:
    if value:
        target.append(str(value).strip())


def _timeline_markdown(timeline: dict[str, Any]) -> str:
    lines = [
        "# Timeline Summary",
        "",
        "| Time | Transcript | OCR text | Visual description |",
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
    lines = ["# Chapter Summaries", ""]
    if not chapters:
        lines.extend(["> No chapter evidence available.", ""])
        return "\n".join(lines)

    for chapter in chapters:
        lines.extend(
            [
                f"## Chapter {chapter['chapter_id']}: {chapter['start']} - {chapter['end']}s",
                "",
                f"- Transcript: {chapter.get('transcript') or 'none'}",
                f"- OCR text: {chapter.get('ocr_text') or 'none'}",
                f"- Visual description: {chapter.get('visual_text') or 'none'}",
                "",
            ]
        )
    return "\n".join(lines)


def _fallback_markdown(evidence: SummaryEvidence, reason: str) -> str:
    lines = [
        "# Video Summary",
        "",
        f"> Final LLM summary did not run successfully: {reason}",
        "",
        "## Selected Evidence",
        "",
    ]
    if evidence.chapters:
        for chapter in evidence.chapters:
            lines.append(f"- Chapter {chapter.chapter_id} ({chapter.start} - {chapter.end}s): selected={len(chapter.selected_events)}")
    else:
        lines.append("- No selected evidence.")
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    return (str(value or "")).replace("|", "\\|").replace("\n", " ")
