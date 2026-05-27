from video_summarizer.config import SummaryConfig
from video_summarizer.summarize import build_chapters, summarize_timeline


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


def test_summarize_without_api_key_writes_all_markdown_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    timeline = {
        "events": [
            {
                "start": 0,
                "end": 30,
                "transcript": "spoken words",
                "ocr_text": "visible title",
                "visual_text": "a user opens the app",
            }
        ]
    }
    final_path = tmp_path / "final_summary.md"
    timeline_path = tmp_path / "timeline_summary.md"
    chapters_path = tmp_path / "chapter_summaries.md"

    status = summarize_timeline(
        timeline,
        final_path,
        SummaryConfig(chapter_window_sec=120),
        timeline_summary_path=timeline_path,
        chapter_summaries_path=chapters_path,
    )

    assert status["status"] == "skipped"
    assert status["reason"] == "missing_openai_api_key"
    assert "a user opens the app" in timeline_path.read_text(encoding="utf-8")
    assert "visible title" in chapters_path.read_text(encoding="utf-8")
    assert "自动最终总结未执行" in final_path.read_text(encoding="utf-8")


def test_summarize_skip_still_writes_layered_reports(tmp_path):
    timeline = {"events": [{"start": 0, "end": 30, "transcript": "", "ocr_text": "", "visual_text": ""}]}
    final_path = tmp_path / "final_summary.md"
    timeline_path = tmp_path / "timeline_summary.md"
    chapters_path = tmp_path / "chapter_summaries.md"

    status = summarize_timeline(
        timeline,
        final_path,
        SummaryConfig(),
        skip=True,
        timeline_summary_path=timeline_path,
        chapter_summaries_path=chapters_path,
    )

    assert status["status"] == "skipped"
    assert timeline_path.exists()
    assert chapters_path.exists()
    assert final_path.exists()
