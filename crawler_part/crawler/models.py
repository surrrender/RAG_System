from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class PageRecord:
    doc_id: str
    url: str
    title: str
    nav_path: list[str]
    raw_text: str
    code_blocks: list[str]
    source: str
    fetched_at: str
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureRecord:
    url: str
    stage: str
    error: str
    retry_count: int
    failed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExtractedPage:
    url: str
    title: str
    nav_path: list[str]
    updated_at: str | None
    raw_text: str
    code_blocks: list[str]
