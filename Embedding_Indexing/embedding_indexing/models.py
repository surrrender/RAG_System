from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    nav_path: list[str]
    section_path: list[str]
    chunk_text: str
    code_blocks: list[str]
    token_estimate: int
    fetched_at: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            url=str(payload["url"]),
            title=str(payload["title"]),
            nav_path=[str(item) for item in payload.get("nav_path", [])],
            section_path=[str(item) for item in payload.get("section_path", [])],
            chunk_text=str(payload["chunk_text"]),
            code_blocks=[str(item) for item in payload.get("code_blocks", [])],
            token_estimate=int(payload.get("token_estimate", 0)),
            fetched_at=str(payload["fetched_at"]),
        )
