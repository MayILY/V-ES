from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LlmCacheConfig, LlmProviderConfig
from .io_utils import read_json, write_json


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    key: str
    path: Path
    text: str | None = None


def provider_cache_params(provider: LlmProviderConfig | None) -> dict[str, Any]:
    if provider is None:
        return {"temperature": None, "max_tokens": None, "top_p": None, "reasoning_effort": None, "extra_body": {}}
    return {
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
        "top_p": provider.top_p,
        "reasoning_effort": provider.reasoning_effort,
        "extra_body": provider.extra_body,
    }


def cache_key(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def lookup_text_cache(config: LlmCacheConfig, namespace: str, key: str) -> CacheLookup:
    path = _cache_path(config.dir, namespace, key)
    if config.mode in {"off", "refresh"}:
        return CacheLookup(False, key, path)
    if not path.exists():
        return CacheLookup(False, key, path)
    data = read_json(path)
    return CacheLookup(True, key, path, str(data.get("text", "")))


def write_text_cache(config: LlmCacheConfig, namespace: str, key: str, text: str, metadata: dict[str, Any]) -> Path | None:
    if config.mode == "off":
        return None
    path = _cache_path(config.dir, namespace, key)
    write_json(
        path,
        {
            "schema_version": config.schema_version,
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "metadata": _plain(metadata),
        },
    )
    return path


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_path(root: Path, namespace: str, key: str) -> Path:
    return root / namespace / f"{key}.json"


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value
