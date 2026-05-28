import json

import pytest

from video_summarizer.config import EvidenceConfig, LlmCacheConfig, LlmConfig, LlmProviderConfig, SummaryConfig
from video_summarizer.evidence import build_summary_evidence
from video_summarizer.summarize import _build_prompt, build_chapters, summarize_timeline


class FakeLlmClient:
    provider_name = "fake"
    model = "fake-model"
    supports_text = True
    supports_vision = False

    def __init__(self, text="generated summary"):
        self.text = text
        self.calls = 0

    def generate_text(self, prompt: str) -> str:
        assert "structured evidence" in prompt
        self.calls += 1
        return self.text

    def describe_image(self, image_path, prompt: str, detail: str = "low", image_data=None) -> str:
        raise NotImplementedError


def test_build_chapters_groups_transcript_ocr_and_visual_evidence():
    timeline = {
        "events": [
            {"start": 0, "end": 30, "transcript": "hello", "ocr_text": "title", "visual_text": "screen"},
            {"start": 60, "end": 90, "transcript": "next", "ocr_text": "", "visual_text": "settings"},
        ]
    }

    chapters = build_chapters(timeline, chapter_window_sec=120)

    assert len(chapters) == 1
    assert chapters[0]["transcript"] == "hello next"
    assert chapters[0]["ocr_text"] == "title"
    assert chapters[0]["visual_text"] == "screen / settings"


def test_summarize_without_available_provider_writes_fallback_and_fails(tmp_path):
    timeline = {"events": [{"start": 0, "end": 30, "transcript": "spoken words", "ocr_text": "visible title", "visual_text": "a user opens the app"}]}

    status = summarize_timeline(
        timeline,
        tmp_path / "final_summary.md",
        SummaryConfig(provider="unknown", model="missing", chapter_window_sec=120),
        timeline_summary_path=tmp_path / "timeline_summary.md",
        chapter_summaries_path=tmp_path / "chapter_summaries.md",
        summary_evidence_json_path=tmp_path / "summary_evidence.json",
        summary_evidence_md_path=tmp_path / "summary_evidence.md",
    )

    assert status["status"] == "failed"
    assert status["reason"] == "unknown_provider: unknown"
    assert "a user opens the app" in (tmp_path / "timeline_summary.md").read_text(encoding="utf-8")
    assert "visible title" in (tmp_path / "chapter_summaries.md").read_text(encoding="utf-8")
    assert "Final LLM summary did not run successfully" in (tmp_path / "final_summary.md").read_text(encoding="utf-8")
    report = (tmp_path / "summary_evidence.md").read_text(encoding="utf-8")
    assert "raw_event_count" in report
    assert "selected_event_count" in report


def test_summarize_uses_injected_llm_client(tmp_path):
    timeline = {"events": [{"start": 0, "end": 30, "transcript": "spoken words", "ocr_text": "", "visual_text": ""}]}
    llm_config = LlmConfig(providers={"fake": LlmProviderConfig(type="custom", api_key_required=False, supports_text=True)})

    status = summarize_timeline(
        timeline,
        tmp_path / "final_summary.md",
        SummaryConfig(provider="fake", model="fake-model"),
        llm_config=llm_config,
        llm_cache_config=LlmCacheConfig(mode="off", dir=tmp_path / "cache"),
        llm_client=FakeLlmClient("generated summary"),
    )

    assert status["status"] == "ok"
    assert status["provider"] == "fake"
    assert status["model"] == "fake-model"
    assert (tmp_path / "final_summary.md").read_text(encoding="utf-8").strip() == "generated summary"


def test_summarize_cache_read_write_skips_second_model_call(tmp_path):
    timeline = {"events": [{"start": 0, "end": 30, "transcript": "spoken words", "ocr_text": "", "visual_text": ""}]}
    llm_config = LlmConfig(providers={"fake": LlmProviderConfig(type="custom", api_key_required=False, supports_text=True)})
    cache = LlmCacheConfig(mode="read_write", dir=tmp_path / "cache")
    client = FakeLlmClient("cached summary")

    first = summarize_timeline(
        timeline,
        tmp_path / "first.md",
        SummaryConfig(provider="fake", model="fake-model"),
        llm_config=llm_config,
        llm_cache_config=cache,
        llm_client=client,
    )
    second = summarize_timeline(
        timeline,
        tmp_path / "second.md",
        SummaryConfig(provider="fake", model="fake-model"),
        llm_config=llm_config,
        llm_cache_config=cache,
        llm_client=client,
    )

    assert first["cache_miss"] is True
    assert first["cache_write"] is True
    assert second["cache_hit"] is True
    assert second["called"] is False
    assert client.calls == 1


def test_summarize_refresh_cache_forces_model_call(tmp_path):
    timeline = {"events": [{"start": 0, "end": 30, "transcript": "spoken words", "ocr_text": "", "visual_text": ""}]}
    llm_config = LlmConfig(providers={"fake": LlmProviderConfig(type="custom", api_key_required=False, supports_text=True)})

    status = summarize_timeline(
        timeline,
        tmp_path / "summary.md",
        SummaryConfig(provider="fake", model="fake-model"),
        llm_config=llm_config,
        llm_cache_config=LlmCacheConfig(mode="refresh", dir=tmp_path / "cache"),
        llm_client=FakeLlmClient("fresh summary"),
    )

    assert status["called"] is True
    assert status["cache_hit"] is False
    assert status["cache_write"] is True


def test_prompt_only_accepts_summary_evidence_and_excludes_raw_fields():
    timeline = {
        "events": [
            {
                "start": 0,
                "end": 30,
                "transcript": "spoken",
                "ocr_text": "title",
                "visual_text": "screen",
                "frames": [{"image_path": "frame_000001.jpg"}],
                "ocr_frames": [{"raw_text": "title"}],
                "transcript_segments": [{"text": "spoken"}],
            }
        ]
    }
    evidence = build_summary_evidence(timeline, EvidenceConfig(), SummaryConfig())
    prompt = _build_prompt(evidence, SummaryConfig())

    assert "frames" not in prompt
    assert "ocr_frames" not in prompt
    assert "transcript_segments" not in prompt
    assert "image_path" not in prompt
    with pytest.raises(TypeError):
        _build_prompt(timeline, SummaryConfig())  # type: ignore[arg-type]


def test_summary_cache_key_changes_when_schema_version_changes(tmp_path):
    timeline = {"events": [{"start": 0, "end": 30, "transcript": "spoken words", "ocr_text": "", "visual_text": ""}]}
    llm_config = LlmConfig(providers={"fake": LlmProviderConfig(type="custom", api_key_required=False, supports_text=True)})
    client = FakeLlmClient("generated")

    first = summarize_timeline(
        timeline,
        tmp_path / "first.md",
        SummaryConfig(provider="fake", model="fake-model"),
        llm_config=llm_config,
        evidence_config=EvidenceConfig(evidence_schema_version="v1"),
        llm_cache_config=LlmCacheConfig(mode="read_write", dir=tmp_path / "cache"),
        llm_client=client,
    )
    second = summarize_timeline(
        timeline,
        tmp_path / "second.md",
        SummaryConfig(provider="fake", model="fake-model"),
        llm_config=llm_config,
        evidence_config=EvidenceConfig(evidence_schema_version="v2"),
        llm_cache_config=LlmCacheConfig(mode="read_write", dir=tmp_path / "cache"),
        llm_client=client,
    )

    assert first["cache_key"] != second["cache_key"]
    assert client.calls == 2
