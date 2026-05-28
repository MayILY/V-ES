from pathlib import Path

from video_summarizer.config import LlmCacheConfig, LlmConfig, LlmProviderConfig, VisionConfig
from video_summarizer.vision import describe_keyframes


class FakeVisionClient:
    provider_name = "fake"
    model = "fake-vision"
    supports_text = True
    supports_vision = True

    def __init__(self):
        self.calls = 0

    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    def describe_image(self, image_path: Path, prompt: str, detail: str = "low", image_data=None) -> str:
        assert "single video frame" in prompt
        self.calls += 1
        return "frame description"


def test_vision_disabled_skips():
    result = describe_keyframes(
        {"scenes": [{"scene_id": 1, "keyframes": [{"frame_id": "f1", "image_path": "f1.jpg"}]}]},
        VisionConfig(enabled=False),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "vision_disabled"
    assert result["enabled"] is False
    assert result["called"] is False
    assert result["cache_hit_count"] == 0


def test_vision_provider_without_image_support_skips_with_frame_placeholders():
    llm_config = LlmConfig(
        providers={
            "text_only": LlmProviderConfig(
                type="custom",
                api_key_required=False,
                supports_text=True,
                supports_vision=False,
            )
        }
    )

    result = describe_keyframes(
        {"scenes": [{"scene_id": 1, "keyframes": [{"frame_id": "f1", "image_path": "f1.jpg"}]}]},
        VisionConfig(enabled=True, provider="text_only", model="text-model"),
        llm_config,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "provider_vision_unsupported"
    assert result["provider"] == "text_only"
    assert result["model"] == "text-model"
    assert result["frames"][0]["status"] == "skipped"


def test_vision_uses_injected_vision_client(tmp_path):
    image = tmp_path / "f1.jpg"
    image.write_bytes(b"fake-image")
    llm_config = LlmConfig(
        providers={
            "fake": LlmProviderConfig(
                type="custom",
                api_key_required=False,
                supports_text=True,
                supports_vision=True,
            )
        }
    )

    result = describe_keyframes(
        {"scenes": [{"scene_id": 1, "keyframes": [{"frame_id": "f1", "image_path": str(image)}]}]},
        VisionConfig(enabled=True, provider="fake", model="fake-vision"),
        llm_config,
        llm_client=FakeVisionClient(),
    )

    assert result["status"] == "ok"
    assert result["called"] is True
    assert result["call_count"] == 1
    assert result["frames"][0]["status"] == "ok"
    assert result["frames"][0]["description"] == "frame description"


def test_vision_cache_partial_hits_only_call_missing_frames(tmp_path):
    image1 = tmp_path / "f1.jpg"
    image2 = tmp_path / "f2.jpg"
    image1.write_bytes(b"fake-image-1")
    image2.write_bytes(b"fake-image-2")
    llm_config = LlmConfig(
        providers={
            "fake": LlmProviderConfig(
                type="custom",
                api_key_required=False,
                supports_text=True,
                supports_vision=True,
            )
        }
    )
    cache = LlmCacheConfig(mode="read_write", dir=tmp_path / "cache")
    client = FakeVisionClient()
    keyframes = {
        "scenes": [
            {
                "scene_id": 1,
                "keyframes": [
                    {"frame_id": "f1", "image_path": str(image1)},
                    {"frame_id": "f2", "image_path": str(image2)},
                ],
            }
        ]
    }

    first = describe_keyframes(keyframes, VisionConfig(enabled=True, provider="fake", model="fake-vision"), llm_config, cache, client)
    second = describe_keyframes(keyframes, VisionConfig(enabled=True, provider="fake", model="fake-vision"), llm_config, cache, client)

    assert first["cache_miss_count"] == 2
    assert first["cache_write_count"] == 2
    assert second["cache_hit_count"] == 2
    assert second["called"] is False
    assert client.calls == 2


def test_vision_refresh_cache_forces_calls(tmp_path):
    image = tmp_path / "f1.jpg"
    image.write_bytes(b"fake-image")
    llm_config = LlmConfig(providers={"fake": LlmProviderConfig(type="custom", api_key_required=False, supports_text=True, supports_vision=True)})
    client = FakeVisionClient()

    result = describe_keyframes(
        {"scenes": [{"scene_id": 1, "keyframes": [{"frame_id": "f1", "image_path": str(image)}]}]},
        VisionConfig(enabled=True, provider="fake", model="fake-vision"),
        llm_config,
        LlmCacheConfig(mode="refresh", dir=tmp_path / "cache"),
        client,
    )

    assert result["called"] is True
    assert result["cache_hit_count"] == 0
    assert result["cache_write_count"] == 1
