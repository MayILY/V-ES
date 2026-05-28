from video_summarizer.config import EvidenceConfig, SummaryConfig
from video_summarizer.evidence import build_summary_evidence


def test_evidence_forces_boundaries_strong_ocr_and_visual_change():
    timeline = {
        "events": [
            {"start": 0, "end": 10, "transcript": "intro", "ocr_text": "", "visual_text": ""},
            {"start": 10, "end": 20, "transcript": "", "ocr_text": "Invoice total 12345 visible title", "visual_text": ""},
            {"start": 20, "end": 30, "transcript": "", "ocr_text": "", "visual_text": "screen change to settings page"},
            {"start": 30, "end": 40, "transcript": "ending", "ocr_text": "", "visual_text": ""},
        ]
    }

    evidence = build_summary_evidence(timeline, EvidenceConfig(), SummaryConfig(chapter_window_sec=120))
    selected = evidence.chapters[0].selected_events
    reasons = {reason for event in selected for reason in event.keep_reason}

    assert "chapter_start" in reasons
    assert "chapter_end" in reasons
    assert "strong_ocr" in reasons
    assert "visual_change" in reasons


def test_dynamic_candidate_count_increases_with_density():
    sparse = {"events": [{"start": i * 60, "end": i * 60 + 30, "transcript": "short", "ocr_text": "", "visual_text": ""} for i in range(4)]}
    dense = {
        "events": [
            {
                "start": i * 60,
                "end": i * 60 + 30,
                "transcript": "detailed spoken explanation " * 3,
                "ocr_text": f"important label {i}",
                "visual_text": f"screen change {i}",
            }
            for i in range(8)
        ]
    }

    sparse_evidence = build_summary_evidence(sparse, EvidenceConfig(), SummaryConfig(chapter_window_sec=600))
    dense_evidence = build_summary_evidence(dense, EvidenceConfig(), SummaryConfig(chapter_window_sec=600))

    assert dense_evidence.chapters[0].stats["dynamic_candidate_count"] > sparse_evidence.chapters[0].stats["dynamic_candidate_count"]


def test_numeric_differences_are_not_deduplicated():
    timeline = {
        "events": [
            {"start": 0, "end": 10, "transcript": "", "ocr_text": "version 1.0 price 100", "visual_text": "screen"},
            {"start": 10, "end": 20, "transcript": "", "ocr_text": "version 2.0 price 200", "visual_text": "screen"},
        ]
    }

    evidence = build_summary_evidence(timeline, EvidenceConfig(event_duplicate_similarity_threshold=0.5), SummaryConfig(chapter_window_sec=120))

    assert len(evidence.chapters[0].selected_events) == 2
    assert evidence.stats["deduplicated_event_count"] == 0


def test_cross_chapter_duplicates_are_downgraded_not_deleted():
    timeline = {
        "events": [
            {"start": 0, "end": 10, "transcript": "same explanation repeated", "ocr_text": "", "visual_text": ""},
            {"start": 130, "end": 140, "transcript": "same explanation repeated", "ocr_text": "", "visual_text": ""},
        ]
    }

    evidence = build_summary_evidence(timeline, EvidenceConfig(event_duplicate_similarity_threshold=0.8), SummaryConfig(chapter_window_sec=120))

    assert len(evidence.chapters) == 2
    assert evidence.chapters[1].selected_events
    assert "cross_chapter_similar_downgraded" in evidence.chapters[1].selected_events[0].keep_reason


def test_same_ocr_with_different_visual_is_kept():
    timeline = {
        "events": [
            {"start": 0, "end": 10, "transcript": "", "ocr_text": "same title", "visual_text": "screen A"},
            {"start": 10, "end": 20, "transcript": "", "ocr_text": "same title", "visual_text": "screen B"},
        ]
    }

    evidence = build_summary_evidence(timeline, EvidenceConfig(event_duplicate_similarity_threshold=0.5), SummaryConfig(chapter_window_sec=120))

    assert len(evidence.chapters[0].selected_events) == 2


def test_prompt_budget_trims_and_records_status():
    timeline = {
        "events": [
            {"start": i * 10, "end": i * 10 + 5, "transcript": "long transcript " * 20, "ocr_text": f"label {i}", "visual_text": "screen change"}
            for i in range(12)
        ]
    }

    evidence = build_summary_evidence(timeline, EvidenceConfig(max_prompt_chars=500), SummaryConfig(chapter_window_sec=300))

    assert evidence.stats["global_trimmed"] is True
    assert evidence.stats["global_trimmed_event_count"] > 0
