from __future__ import annotations

from pathlib import Path

import orjson

from embedding_indexing.models import ChunkRecord


def iter_chunks(path: Path):
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number} in {path}") from exc
            yield ChunkRecord.from_dict(payload) # TODO:同样这里采用了逐步返回的策略


def load_chunks(path: Path) -> list[ChunkRecord]:
    return list(iter_chunks(path))
