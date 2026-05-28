from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import LlmConfig, LlmProviderConfig


class LlmClient(Protocol):
    provider_name: str
    model: str
    supports_text: bool
    supports_vision: bool

    def generate_text(self, prompt: str) -> str:
        ...

    def describe_image(self, image_path: Path, prompt: str, detail: str = "low", image_data: bytes | None = None) -> str:
        ...


class LlmProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderReadiness:
    provider: str
    status: str
    reason: str | None = None
    suggestion: str = ""


ModuleFinder = Callable[[str], object | None]


class OpenAICompatibleClient:
    def __init__(self, provider_name: str, provider: LlmProviderConfig, model: str, api_key: str | None) -> None:
        self.provider_name = provider_name
        self.provider = provider
        self.model = model
        self.api_key = api_key or "local-not-required"
        self.supports_text = provider.supports_text
        self.supports_vision = provider.supports_vision

    def generate_text(self, prompt: str) -> str:
        client = self._client()
        response = client.chat.completions.create(**self._chat_completion_kwargs([{"role": "user", "content": prompt}]))
        text = response.choices[0].message.content
        return str(text or "").strip()

    def describe_image(self, image_path: Path, prompt: str, detail: str = "low", image_data: bytes | None = None) -> str:
        if not self.supports_vision:
            raise LlmProviderError("provider_vision_unsupported")

        data_url = image_to_data_url(image_path, image_data)
        client = self._client()
        if self.provider.vision_format == "responses":
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url, "detail": detail},
                        ],
                    }
                ],
            )
            return _extract_response_text(response).strip()

        if self.provider.vision_format == "chat_completions":
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
                        ],
                    }
                ],
                stream=False,
            )
            text = response.choices[0].message.content
            return str(text or "").strip()

        raise LlmProviderError("provider_vision_format_unsupported")

    def _client(self) -> Any:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise LlmProviderError(f"provider_package_unavailable: openai: {exc}") from exc

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.provider.base_url:
            kwargs["base_url"] = self.provider.base_url
        return OpenAI(**kwargs)

    def _chat_completion_kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.provider.reasoning_effort:
            kwargs["reasoning_effort"] = self.provider.reasoning_effort
        if self.provider.temperature is not None:
            kwargs["temperature"] = self.provider.temperature
        if self.provider.max_tokens is not None:
            kwargs["max_tokens"] = self.provider.max_tokens
        if self.provider.top_p is not None:
            kwargs["top_p"] = self.provider.top_p
        if self.provider.extra_body:
            kwargs["extra_body"] = self.provider.extra_body
        return kwargs


class GeminiClient:
    def __init__(self, provider_name: str, provider: LlmProviderConfig, model: str, api_key: str | None) -> None:
        self.provider_name = provider_name
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.supports_text = provider.supports_text
        self.supports_vision = provider.supports_vision

    def generate_text(self, prompt: str) -> str:
        try:
            from google import genai  # type: ignore
        except Exception as exc:
            raise LlmProviderError(f"provider_package_unavailable: google-genai: {exc}") from exc

        kwargs = {"api_key": self.api_key} if self.api_key else {}
        client = genai.Client(**kwargs)
        response = client.models.generate_content(model=self.model, contents=prompt)
        return str(getattr(response, "text", "") or "").strip()

    def describe_image(self, image_path: Path, prompt: str, detail: str = "low", image_data: bytes | None = None) -> str:
        raise LlmProviderError("provider_vision_unsupported")


def create_llm_client(
    provider_name: str,
    model: str,
    llm_config: LlmConfig,
    env: dict[str, str] | None = None,
) -> LlmClient:
    env = env if env is not None else os.environ
    provider_key = provider_name.lower().strip()
    provider = llm_config.providers.get(provider_key)
    if provider is None:
        raise LlmProviderError(f"unknown_provider: {provider_name}")

    if not model:
        raise LlmProviderError(f"missing_model: {provider_key}")

    api_key = env.get(provider.api_key_env) if provider.api_key_env else None
    if provider.api_key_required and not api_key:
        raise LlmProviderError(f"missing_api_key: {provider.api_key_env}")

    if provider.type == "openai_compatible":
        return OpenAICompatibleClient(provider_key, provider, model, api_key)
    if provider.type == "gemini":
        return GeminiClient(provider_key, provider, model, api_key)
    raise LlmProviderError(f"unsupported_provider_type: {provider.type}")


def check_provider_ready(
    provider_name: str,
    model: str,
    llm_config: LlmConfig,
    *,
    need_vision: bool = False,
    env: dict[str, str] | None = None,
    module_finder: ModuleFinder | None = None,
) -> ProviderReadiness:
    env = env if env is not None else os.environ
    module_finder = module_finder or importlib.util.find_spec
    provider_key = provider_name.lower().strip()
    provider = llm_config.providers.get(provider_key)
    if provider is None:
        return ProviderReadiness(provider_key, "missing", f"unknown_provider: {provider_name}")

    if not model:
        return ProviderReadiness(provider_key, "missing", f"missing_model: {provider_key}")

    package_name = _provider_package(provider)
    try:
        package_missing = bool(package_name and module_finder(package_name) is None)
    except Exception:
        package_missing = True
    if package_missing and package_name:
        return ProviderReadiness(provider_key, "missing", f"provider_package_unavailable: {package_name}", _package_suggestion(provider))

    if provider.api_key_required and provider.api_key_env and not env.get(provider.api_key_env):
        return ProviderReadiness(provider_key, "missing", f"missing_api_key: {provider.api_key_env}", f"Set {provider.api_key_env}.")

    if need_vision and not provider.supports_vision:
        return ProviderReadiness(provider_key, "warning", "provider_vision_unsupported")

    capability = "vision" if need_vision else "text"
    if capability == "text" and not provider.supports_text:
        return ProviderReadiness(provider_key, "missing", "provider_text_unsupported")

    return ProviderReadiness(provider_key, "ok")


def image_to_data_url(path: Path, data: bytes | None = None) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    payload = data if data is not None else path.read_bytes()
    import base64

    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _provider_package(provider: LlmProviderConfig) -> str | None:
    if provider.type == "openai_compatible":
        return "openai"
    if provider.type == "gemini":
        return "google.genai"
    return None


def _package_suggestion(provider: LlmProviderConfig) -> str:
    if provider.type == "gemini":
        return 'python -m pip install -e ".[gemini]"'
    return 'python -m pip install -e ".[ai]"'


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)
