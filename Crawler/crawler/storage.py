from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson


def ensure_runtime_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")


def load_fingerprints(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_fingerprints(path: Path, fingerprints: dict[str, str]) -> None:
    path.write_text(json.dumps(fingerprints, ensure_ascii=False, indent=2), encoding="utf-8")
