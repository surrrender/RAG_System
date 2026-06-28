from __future__ import annotations

import re

from crawler.models import ChunkRecord, PageRecord
from crawler.utils import estimate_tokens, make_chunk_id


_SEMANTIC_SPLIT_PATTERNS = (
    re.compile(r"\n{2,}"),
    re.compile(r"\n+"),
    re.compile(r"[。！？!?；;]+(?:[\"'”’」』】》）]+)?"),
    re.compile(r"[，、,:：]+(?:[\"'”’」』】》）]+)?"),
    re.compile(r"\s+"),
)


def build_chunks(
    page: PageRecord,
    heading_blocks: list[dict[str, object]],
    max_chars: int = 500,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for block in heading_blocks:
        text = str(block["text"]).strip()
        section_path = [str(item).lstrip('#') for item in block["section_path"]]
        if len(text) < 10:
            continue

        for split_text in _split_text_semantically(text, max_chars=max_chars):
            chunk = _build_chunk(page=page, section_path=section_path, text=split_text)
            if chunk is not None:
                chunks.append(chunk)

    if not chunks:
        for split_text in _split_text_semantically(page.raw_text, max_chars=max_chars):
            chunk = _build_chunk(page=page, section_path=[], text=split_text)
            if chunk is not None:
                chunks.append(chunk)
    return chunks


def _split_text_semantically(text: str, max_chars: int) -> list[str]:
    body = text.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]
    return _split_text_recursively(body, max_chars=max_chars, patterns=_SEMANTIC_SPLIT_PATTERNS)


def _split_text_recursively(text: str, max_chars: int, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    body = text.strip()
    if len(body) <= max_chars:
        return [body]
    if not patterns:
        return _hard_split_text(body, max_chars=max_chars)

    pattern = patterns[0]
    remaining_patterns = patterns[1:]
    segments = _split_by_pattern(body, pattern)
    if len(segments) == 1:
        return _split_text_recursively(body, max_chars=max_chars, patterns=remaining_patterns)

    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current}{segment}"
        if current and len(candidate.strip()) > max_chars:
            chunks.extend(
                _finalize_or_split_chunk(
                    current,
                    max_chars=max_chars,
                    patterns=remaining_patterns,
                )
            )
            current = segment
            continue
        current = candidate

    if current.strip():
        chunks.extend(
            _finalize_or_split_chunk(
                current,
                max_chars=max_chars,
                patterns=remaining_patterns,
            )
        )
    return chunks


def _finalize_or_split_chunk(text: str, max_chars: int, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    body = text.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]
    return _split_text_recursively(body, max_chars=max_chars, patterns=patterns)


def _split_by_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    segments: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        end = match.end()
        segment = text[start:end]
        if segment.strip():
            segments.append(segment)
        start = end

    tail = text[start:]
    if tail.strip():
        segments.append(tail)
    return segments


def _hard_split_text(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars

        chunk = remaining[:split_at].strip()
        if not chunk:
            split_at = max_chars
            chunk = remaining[:split_at].strip()

        chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _build_chunk(page: PageRecord, section_path: list[str], text: str) -> ChunkRecord | None:
    body = text.strip()
    if not body:
        return None
    return ChunkRecord(
        chunk_id=make_chunk_id(page.doc_id, section_path, body, chunk_type="text"),
        doc_id=page.doc_id,
        url=page.url,
        title=page.title,
        nav_path=page.nav_path,
        section_path=section_path,
        chunk_text=body,
        token_estimate=estimate_tokens(body),
        fetched_at=page.fetched_at,
    )
