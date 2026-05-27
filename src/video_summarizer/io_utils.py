from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_stem(path: Path) -> str:
    stem = path.stem.strip().replace(" ", "_")
    keep = []
    for char in stem:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    value = "".join(keep).strip("._-")
    return value or "media"
