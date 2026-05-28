from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .config import EvidenceConfig, SummaryConfig


@dataclass
class SummaryEvidenceEvent:
    start: float
    end: float
    transcript: str
    ocr_text: str
    visual_text: str
    score: float
    score_breakdown: dict[str, float]
    keep_reason: list[str] = field(default_factory=list)
    drop_reason: list[str] = field(default_factory=list)
    forced: bool = False
    selected: bool = False


@dataclass
class SummaryEvidenceChapter:
    chapter_id: int
    start: float
    end: float
    selected_events: list[SummaryEvidenceEvent]
    dropped_events: list[SummaryEvidenceEvent]
    stats: dict[str, Any]
    transcript_summary: str = ""
    ocr_summary: str = ""
    visual_summary: str = ""


@dataclass
class SummaryEvidence:
    schema_version: str
    builder_version: str
    prompt_template_version: str
    output_language: str
    summary_style: str
    chapters: list[SummaryEvidenceChapter]
    stats: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_summary_evidence(
    timeline: dict[str, Any],
    evidence_config: EvidenceConfig,
    summary_config: SummaryConfig,
) -> SummaryEvidence:
    raw_events = list(timeline.get("events", []))
    chapters_raw = _split_chapters(raw_events, summary_config.chapter_window_sec, float(timeline.get("window_sec", 60.0)))
    chapters: list[SummaryEvidenceChapter] = []
    totals = {
        "input_event_count": len(raw_events),
        "selected_event_count": 0,
        "dropped_event_count": 0,
        "deduplicated_event_count": 0,
        "low_info_filtered_count": 0,
        "global_trimmed_event_count": 0,
        "global_trimmed": False,
        "trim_reasons": [],
        "text_chars_before": {"transcript": 0, "ocr_text": 0, "visual_text": 0},
        "text_chars_after": {"transcript": 0, "ocr_text": 0, "visual_text": 0},
    }
    seen_cross_chapter: list[SummaryEvidenceEvent] = []

    for index, chapter_events in enumerate(chapters_raw, start=1):
        chapter = _build_chapter(chapter_events, index, evidence_config, seen_cross_chapter, totals)
        chapters.append(chapter)
        seen_cross_chapter.extend(chapter.selected_events)

    evidence = SummaryEvidence(
        schema_version=evidence_config.evidence_schema_version,
        builder_version=evidence_config.builder_version,
        prompt_template_version=evidence_config.prompt_template_version,
        output_language=summary_config.output_language,
        summary_style=summary_config.summary_style,
        chapters=chapters,
        stats=totals,
        config=_config_snapshot(evidence_config, summary_config),
    )
    _apply_global_prompt_budget(evidence, evidence_config.max_prompt_chars)
    evidence.stats["selected_event_count"] = sum(len(chapter.selected_events) for chapter in evidence.chapters)
    evidence.stats["dropped_event_count"] = sum(len(chapter.dropped_events) for chapter in evidence.chapters)
    evidence.stats["estimated_prompt_chars"] = estimate_prompt_chars(evidence)
    return evidence


def estimate_prompt_chars(evidence: SummaryEvidence) -> int:
    total = 260
    for chapter in evidence.chapters:
        total += 80
        total += len(chapter.transcript_summary) + len(chapter.ocr_summary) + len(chapter.visual_summary)
        for event in chapter.selected_events:
            total += 80 + len(event.transcript) + len(event.ocr_text) + len(event.visual_text)
    return total


def evidence_report_markdown(evidence: SummaryEvidence) -> str:
    stats = evidence.stats
    lines = [
        "# Summary Evidence Acceptance Report",
        "",
        "## Configuration",
        "",
        f"- evidence_schema_version: {evidence.schema_version}",
        f"- builder_version: {evidence.builder_version}",
        f"- prompt_template_version: {evidence.prompt_template_version}",
        f"- output_language: {evidence.output_language}",
        f"- summary_style: {evidence.summary_style}",
        f"- llm_cache_mode: {evidence.config.get('llm_cache_mode', 'unknown')}",
        "",
        "## Totals",
        "",
        f"- raw_event_count: {stats.get('input_event_count', 0)}",
        f"- selected_event_count: {stats.get('selected_event_count', 0)}",
        f"- deduplicated_event_count: {stats.get('deduplicated_event_count', 0)}",
        f"- low_info_filtered_count: {stats.get('low_info_filtered_count', 0)}",
        f"- global_trimmed_event_count: {stats.get('global_trimmed_event_count', 0)}",
        f"- estimated_prompt_chars: {stats.get('estimated_prompt_chars', 0)}",
        f"- global_trimmed: {stats.get('global_trimmed', False)}",
        f"- text_chars_before: {stats.get('text_chars_before', {})}",
        f"- text_chars_after: {stats.get('text_chars_after', {})}",
        "",
        "## Chapters",
        "",
    ]
    for chapter in evidence.chapters:
        keep_reasons = Counter(reason for event in chapter.selected_events for reason in event.keep_reason)
        drop_reasons = Counter(reason for event in chapter.dropped_events for reason in event.drop_reason)
        lines.extend(
            [
                f"### Chapter {chapter.chapter_id}: {chapter.start} - {chapter.end}s",
                "",
                f"- selected: {len(chapter.selected_events)}",
                f"- dropped: {len(chapter.dropped_events)}",
                f"- keep_reasons: {dict(keep_reasons)}",
                f"- drop_reasons: {dict(drop_reasons)}",
                f"- stats: {chapter.stats}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _split_chapters(events: list[dict[str, Any]], chapter_window_sec: float, fallback_window: float) -> list[list[dict[str, Any]]]:
    if not events:
        return []
    chapter_window_sec = chapter_window_sec if chapter_window_sec > 0 else fallback_window
    chapters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chapter_start = float(events[0].get("start", 0.0))
    for event in events:
        start = float(event.get("start", 0.0))
        if current and start >= chapter_start + chapter_window_sec:
            chapters.append(current)
            current = []
            chapter_start = start
        current.append(event)
    if current:
        chapters.append(current)
    return chapters


def _build_chapter(
    raw_events: list[dict[str, Any]],
    chapter_id: int,
    config: EvidenceConfig,
    seen_cross_chapter: list[SummaryEvidenceEvent],
    totals: dict[str, Any],
) -> SummaryEvidenceChapter:
    scored = [_score_event(event, idx, len(raw_events), config, totals) for idx, event in enumerate(raw_events)]
    _apply_duplicate_rules(scored, config, seen_cross_chapter, totals)
    candidate_count = _dynamic_candidate_count(scored)
    forced = [event for event in scored if event.forced and not _is_dropped(event)]
    regular = [event for event in scored if event not in forced and not _is_dropped(event)]
    regular.sort(key=lambda event: event.score, reverse=True)
    selected = sorted((forced + regular)[: max(candidate_count, len(forced))], key=lambda event: event.start)
    dropped = []
    for event in scored:
        if event in selected:
            event.selected = True
            continue
        if not event.drop_reason:
            event.drop_reason.append("below_dynamic_candidate_cutoff")
        dropped.append(event)

    for event in selected:
        _compress_event(event, config, totals)
    chapter = SummaryEvidenceChapter(
        chapter_id=chapter_id,
        start=round(float(raw_events[0].get("start", 0.0)), 3),
        end=round(float(raw_events[-1].get("end", raw_events[-1].get("start", 0.0))), 3),
        selected_events=selected,
        dropped_events=dropped,
        stats={
            "raw_event_count": len(raw_events),
            "selected_event_count": len(selected),
            "dropped_event_count": len(dropped),
            "dynamic_candidate_count": candidate_count,
        },
    )
    chapter.transcript_summary = _truncate(" ".join(event.transcript for event in selected if event.transcript), config.max_chapter_summary_chars)
    chapter.ocr_summary = _truncate(" / ".join(event.ocr_text for event in selected if event.ocr_text), config.max_chapter_summary_chars)
    chapter.visual_summary = _truncate(" / ".join(event.visual_text for event in selected if event.visual_text), config.max_chapter_summary_chars)
    return chapter


def _score_event(
    event: dict[str, Any],
    index: int,
    chapter_size: int,
    config: EvidenceConfig,
    totals: dict[str, Any],
) -> SummaryEvidenceEvent:
    transcript = str(event.get("transcript") or "").strip()
    ocr_text = str(event.get("ocr_text") or "").strip()
    visual_text = str(event.get("visual_text") or "").strip()
    for key, value in {"transcript": transcript, "ocr_text": ocr_text, "visual_text": visual_text}.items():
        totals["text_chars_before"][key] += len(value)
    breakdown = {
        "ocr_strength": min(len(_normalize(ocr_text)) / 40.0, 3.0),
        "visual_strength": min(len(_normalize(visual_text)) / 50.0, 2.5),
        "transcript_info": min(len(_normalize(transcript)) / 80.0, 2.0),
        "chapter_boundary": 0.0,
        "scene_change": 0.0,
        "novelty": 1.0,
        "keyword_signal": _keyword_score(" ".join([transcript, ocr_text, visual_text]), config.keyword_signal_terms),
        "low_info_penalty": 0.0,
        "duplicate_penalty": 0.0,
    }
    keep_reason: list[str] = []
    forced = False
    if config.preserve_chapter_boundaries and index == 0:
        breakdown["chapter_boundary"] += 2.0
        keep_reason.append("chapter_start")
        forced = True
    if config.preserve_chapter_boundaries and index == chapter_size - 1:
        breakdown["chapter_boundary"] += 2.0
        keep_reason.append("chapter_end")
        forced = True
    if len(_normalize(ocr_text)) >= max(config.min_text_info_chars * 2, 24):
        keep_reason.append("strong_ocr")
        forced = True
    if visual_text and ("change" in visual_text.lower() or "screen" in visual_text.lower() or "界面" in visual_text or "切换" in visual_text):
        breakdown["scene_change"] = 1.5
        keep_reason.append("visual_change")
        forced = True
    normalized_total = _normalize(" ".join([transcript, ocr_text, visual_text]))
    if len(normalized_total) < config.min_text_info_chars:
        breakdown["low_info_penalty"] = -2.0
    score = round(sum(breakdown.values()), 4)
    output = SummaryEvidenceEvent(
        start=round(float(event.get("start", 0.0)), 3),
        end=round(float(event.get("end", event.get("start", 0.0))), 3),
        transcript=transcript,
        ocr_text=ocr_text,
        visual_text=visual_text,
        score=score,
        score_breakdown=breakdown,
        keep_reason=keep_reason,
        forced=forced,
    )
    if breakdown["low_info_penalty"] < 0 and not forced:
        output.drop_reason.append("low_information")
        totals["low_info_filtered_count"] += 1
    return output


def _apply_duplicate_rules(
    events: list[SummaryEvidenceEvent],
    config: EvidenceConfig,
    seen_cross_chapter: list[SummaryEvidenceEvent],
    totals: dict[str, Any],
) -> None:
    kept_in_chapter: list[SummaryEvidenceEvent] = []
    for event in events:
        if event.drop_reason:
            continue
        duplicate = _find_duplicate(event, kept_in_chapter, config)
        if duplicate is not None and not _numeric_difference(event, duplicate, config) and not _ocr_same_visual_different(event, duplicate):
            if event.forced:
                event.score_breakdown["duplicate_penalty"] = -0.5
                event.score = round(event.score - 0.5, 4)
                event.keep_reason.append("forced_duplicate_downgraded")
                kept_in_chapter.append(event)
                continue
            event.drop_reason.append("duplicate_within_chapter")
            totals["deduplicated_event_count"] += 1
            continue
        cross_duplicate = _find_duplicate(event, seen_cross_chapter, config)
        if cross_duplicate is not None and not _numeric_difference(event, cross_duplicate, config):
            event.score_breakdown["duplicate_penalty"] = -0.8
            event.score = round(event.score - 0.8, 4)
            event.keep_reason.append("cross_chapter_similar_downgraded")
        kept_in_chapter.append(event)


def _find_duplicate(
    event: SummaryEvidenceEvent,
    previous: list[SummaryEvidenceEvent],
    config: EvidenceConfig,
) -> SummaryEvidenceEvent | None:
    current = _normalize(" ".join([event.transcript, event.ocr_text, event.visual_text]))
    if not current:
        return None
    for candidate in previous:
        prior = _normalize(" ".join([candidate.transcript, candidate.ocr_text, candidate.visual_text]))
        if not prior:
            continue
        if abs(event.start - candidate.start) > 900:
            continue
        if SequenceMatcher(None, current, prior).ratio() >= config.event_duplicate_similarity_threshold:
            return candidate
    return None


def _dynamic_candidate_count(events: list[SummaryEvidenceEvent]) -> int:
    if not events:
        return 0
    duration = max(event.end for event in events) - min(event.start for event in events)
    density = sum(1 for event in events if event.ocr_text or event.visual_text or len(_normalize(event.transcript)) >= 20)
    return max(2, min(len(events), 2 + int(duration // 180) + int(density // 3)))


def _apply_global_prompt_budget(evidence: SummaryEvidence, max_prompt_chars: int) -> None:
    while max_prompt_chars > 0 and estimate_prompt_chars(evidence) > max_prompt_chars:
        candidates: list[tuple[float, SummaryEvidenceChapter, SummaryEvidenceEvent]] = []
        for chapter in evidence.chapters:
            if len(chapter.selected_events) <= 1:
                continue
            for event in chapter.selected_events:
                priority = event.score + (10.0 if event.forced else 0.0)
                candidates.append((priority, chapter, event))
        if not candidates:
            break
        _, chapter, event = min(candidates, key=lambda item: item[0])
        chapter.selected_events.remove(event)
        event.selected = False
        event.drop_reason.append("global_prompt_budget_trim")
        chapter.dropped_events.append(event)
        evidence.stats["global_trimmed"] = True
        evidence.stats["global_trimmed_event_count"] += 1
        evidence.stats["trim_reasons"].append("global_prompt_budget_trim")


def _compress_event(event: SummaryEvidenceEvent, config: EvidenceConfig, totals: dict[str, Any]) -> None:
    event.transcript = _record_truncate(event.transcript, "transcript", config.max_chapter_summary_chars, totals)
    event.ocr_text = _record_truncate(event.ocr_text, "ocr_text", config.max_chapter_summary_chars, totals)
    event.visual_text = _record_truncate(event.visual_text, "visual_text", config.max_chapter_summary_chars, totals)


def _record_truncate(text: str, key: str, limit: int, totals: dict[str, Any]) -> str:
    value = _truncate(text, limit)
    totals["text_chars_after"][key] += len(value)
    return value


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _keyword_score(text: str, terms: list[str]) -> float:
    lowered = text.lower()
    return min(sum(1 for term in terms if term and term.lower() in lowered) * 0.8, 2.4)


def _numeric_difference(left: SummaryEvidenceEvent, right: SummaryEvidenceEvent, config: EvidenceConfig) -> bool:
    if not config.preserve_numeric_differences:
        return False
    return _numeric_tokens(_all_text(left)) != _numeric_tokens(_all_text(right))


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[./:-]\d+)*(?:\.\d+)?|[$¥￥]\s*\d+(?:\.\d+)?", text))


def _ocr_same_visual_different(left: SummaryEvidenceEvent, right: SummaryEvidenceEvent) -> bool:
    return bool(left.ocr_text and _normalize(left.ocr_text) == _normalize(right.ocr_text) and _normalize(left.visual_text) != _normalize(right.visual_text))


def _all_text(event: SummaryEvidenceEvent) -> str:
    return " ".join([event.transcript, event.ocr_text, event.visual_text])


def _is_dropped(event: SummaryEvidenceEvent) -> bool:
    return any(reason in {"low_information", "duplicate_within_chapter"} for reason in event.drop_reason)


def _config_snapshot(evidence_config: EvidenceConfig, summary_config: SummaryConfig) -> dict[str, Any]:
    return {
        "min_text_info_chars": evidence_config.min_text_info_chars,
        "preserve_chapter_boundaries": evidence_config.preserve_chapter_boundaries,
        "preserve_numeric_differences": evidence_config.preserve_numeric_differences,
        "max_chapter_summary_chars": evidence_config.max_chapter_summary_chars,
        "max_prompt_chars": evidence_config.max_prompt_chars,
        "event_duplicate_similarity_threshold": evidence_config.event_duplicate_similarity_threshold,
        "chapter_window_sec": summary_config.chapter_window_sec,
        "output_language": summary_config.output_language,
        "summary_style": summary_config.summary_style,
    }
