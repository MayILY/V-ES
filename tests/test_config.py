from pathlib import Path

from video_summarizer.config import load_config, load_env_file


def test_load_ocr_ppocrv5_config_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ocr:
  version: PP-OCRv5
  device: cpu
  model_root: D:/someElse/video_summarizer-models/paddleocr
  text_detection_model_name: PP-OCRv5_mobile_det
  text_recognition_model_name: PP-OCRv5_mobile_rec
  use_doc_orientation_classify: false
  use_doc_unwarping: false
  use_textline_orientation: false
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.version == "PP-OCRv5"
    assert config.ocr.device == "cpu"
    assert config.ocr.model_root == Path("D:/someElse/video_summarizer-models/paddleocr")
    assert config.ocr.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert config.ocr.text_recognition_model_name == "PP-OCRv5_mobile_rec"
    assert config.ocr.use_doc_orientation_classify is False
    assert config.ocr.use_doc_unwarping is False
    assert config.ocr.use_textline_orientation is False


def test_load_scene_detection_config_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
scene_detection:
  enabled: true
  detector: adaptive
  threshold: 30.0
  min_scene_len_sec: 2.5
  adaptive_threshold: 4.0
  min_content_val: 12.0
  window_width: 3
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.scene_detection.enabled is True
    assert config.scene_detection.detector == "adaptive"
    assert config.scene_detection.threshold == 30.0
    assert config.scene_detection.min_scene_len_sec == 2.5
    assert config.scene_detection.adaptive_threshold == 4.0
    assert config.scene_detection.min_content_val == 12.0
    assert config.scene_detection.window_width == 3


def test_load_llm_provider_config_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
summary:
  provider: local
  model: qwen2.5:7b
vision:
  provider: qwen
  model: qwen-vl-max
llm:
  providers:
    local:
      base_url: http://127.0.0.1:1234/v1
      api_key_required: false
    qwen:
      supports_vision: true
      vision_format: chat_completions
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.summary.provider == "local"
    assert config.summary.model == "qwen2.5:7b"
    assert config.vision.provider == "qwen"
    assert config.vision.model == "qwen-vl-max"
    assert config.llm.providers["local"].base_url == "http://127.0.0.1:1234/v1"
    assert config.llm.providers["local"].api_key_required is False
    assert config.llm.providers["qwen"].api_key_env == "DASHSCOPE_API_KEY"
    assert config.llm.providers["qwen"].supports_vision is True
    assert config.llm.providers["qwen"].vision_format == "chat_completions"


def test_load_evidence_and_cache_config_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
summary:
  summary_style: bullet
evidence:
  min_text_info_chars: 16
  preserve_chapter_boundaries: true
  preserve_numeric_differences: true
  max_chapter_summary_chars: 800
  max_prompt_chars: 5000
  evidence_schema_version: evidence-v2
  builder_version: builder-v2
  prompt_template_version: prompt-v2
llm_cache:
  mode: refresh
  dir: outputs/custom-cache
  schema_version: cache-v2
llm:
  providers:
    openai:
      temperature: 0.2
      max_tokens: 1200
      top_p: 0.9
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.summary.summary_style == "bullet"
    assert config.evidence.min_text_info_chars == 16
    assert config.evidence.max_prompt_chars == 5000
    assert config.evidence.evidence_schema_version == "evidence-v2"
    assert config.llm_cache.mode == "refresh"
    assert config.llm_cache.dir == Path("outputs/custom-cache")
    assert config.llm_cache.schema_version == "cache-v2"
    assert config.llm.providers["openai"].temperature == 0.2
    assert config.llm.providers["openai"].max_tokens == 1200
    assert config.llm.providers["openai"].top_p == 0.9


def test_default_deepseek_provider_uses_v4_thinking_parameters():
    config = load_config(Path("missing-config.yaml"))

    deepseek = config.llm.providers["deepseek"]
    assert deepseek.reasoning_effort == "max"
    assert deepseek.extra_body == {"thinking": {"type": "enabled"}}


def test_load_env_file_reads_multiple_provider_keys_without_overriding(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        """
# comment
OPENAI_API_KEY=from-file
DEEPSEEK_API_KEY="deepseek-from-file"
DASHSCOPE_API_KEY='qwen-from-file'
GEMINI_API_KEY=gemini-from-file
""",
        encoding="utf-8",
    )
    env = {"OPENAI_API_KEY": "already-set"}

    loaded = load_env_file(env_path, override=False, environ=env)

    assert env["OPENAI_API_KEY"] == "already-set"
    assert env["DEEPSEEK_API_KEY"] == "deepseek-from-file"
    assert env["DASHSCOPE_API_KEY"] == "qwen-from-file"
    assert env["GEMINI_API_KEY"] == "gemini-from-file"
    assert loaded == {
        "DEEPSEEK_API_KEY": "deepseek-from-file",
        "DASHSCOPE_API_KEY": "qwen-from-file",
        "GEMINI_API_KEY": "gemini-from-file",
    }


def test_load_config_reads_config_relative_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".keys").write_text("DEEPSEEK_API_KEY=from-config-file\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
summary:
  provider: deepseek
  model: deepseek-v4-flash
llm:
  env_file: .keys
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.llm.env_file == tmp_path / ".keys"
    assert config.summary.provider == "deepseek"
    assert config.summary.model == "deepseek-v4-flash"
    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "from-config-file"
