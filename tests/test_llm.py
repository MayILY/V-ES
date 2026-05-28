from video_summarizer.config import LlmConfig, LlmProviderConfig
from video_summarizer.llm import (
    GeminiClient,
    OpenAICompatibleClient,
    check_provider_ready,
    create_llm_client,
)


def test_openai_compatible_factory_covers_cloud_and_local_providers():
    config = LlmConfig()

    openai_client = create_llm_client("openai", "gpt-4.1-mini", config, env={"OPENAI_API_KEY": "key"})
    deepseek_client = create_llm_client("deepseek", "deepseek-v4-pro", config, env={"DEEPSEEK_API_KEY": "key"})
    qwen_client = create_llm_client("qwen", "qwen-plus", config, env={"DASHSCOPE_API_KEY": "key"})
    local_client = create_llm_client("local", "qwen2.5:7b", config, env={})

    assert isinstance(openai_client, OpenAICompatibleClient)
    assert isinstance(deepseek_client, OpenAICompatibleClient)
    assert isinstance(qwen_client, OpenAICompatibleClient)
    assert isinstance(local_client, OpenAICompatibleClient)
    assert local_client.provider.base_url == "http://localhost:11434/v1"


def test_deepseek_client_sends_v4_thinking_parameters():
    config = LlmConfig()
    client = create_llm_client("deepseek", "deepseek-v4-pro", config, env={"DEEPSEEK_API_KEY": "key"})

    kwargs = client._chat_completion_kwargs([{"role": "user", "content": "summarize"}])

    assert kwargs["model"] == "deepseek-v4-pro"
    assert kwargs["reasoning_effort"] == "max"
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_gemini_factory_uses_gemini_client():
    client = create_llm_client("gemini", "gemini-3.5-flash", LlmConfig(), env={"GEMINI_API_KEY": "key"})

    assert isinstance(client, GeminiClient)


def test_provider_check_reports_missing_cloud_key_but_not_local_key():
    module_finder = lambda name: object()

    cloud = check_provider_ready("openai", "gpt-4.1-mini", LlmConfig(), env={}, module_finder=module_finder)
    local = check_provider_ready("local", "qwen2.5:7b", LlmConfig(), env={}, module_finder=module_finder)

    assert cloud.status == "missing"
    assert cloud.reason == "missing_api_key: OPENAI_API_KEY"
    assert local.status == "ok"


def test_provider_check_reports_vision_unsupported():
    config = LlmConfig(
        providers={
            "text_only": LlmProviderConfig(
                type="custom",
                api_key_required=False,
                supports_text=True,
                supports_vision=False,
            )
        }
    )

    result = check_provider_ready("text_only", "model", config, need_vision=True, env={})

    assert result.status == "warning"
    assert result.reason == "provider_vision_unsupported"
