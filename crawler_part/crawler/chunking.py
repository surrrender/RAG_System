from __future__ import annotations

from crawler.models import ChunkRecord, PageRecord
from crawler.utils import estimate_tokens, make_chunk_id


def build_chunks(page: PageRecord, heading_blocks: list[dict[str, object]], min_chars: int = 80) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for block in heading_blocks:
        text = str(block["text"]).strip()
        codes = [str(item) for item in block["code_blocks"]]
        section_path = [str(item) for item in block["section_path"]]
        if not text and not codes:
            continue

        if chunks and len(text) < min_chars:
            previous = chunks[-1]
            merged_text = f"{previous.chunk_text}\n\n{text}".strip() if text else previous.chunk_text
            previous.chunk_text = merged_text
            previous.code_blocks.extend(codes)
            previous.token_estimate = estimate_tokens(merged_text)
            continue

        body = text or "\n".join(codes)
        chunks.append(
            ChunkRecord(
                chunk_id=make_chunk_id(page.doc_id, section_path, body),
                doc_id=page.doc_id,
                url=page.url,
                title=page.title,
                nav_path=page.nav_path,
                section_path=section_path,
                chunk_text=body,
                code_blocks=codes,
                token_estimate=estimate_tokens(body),
                fetched_at=page.fetched_at,
            )
        )

    if not chunks:
        chunks.append(
            ChunkRecord(
                chunk_id=make_chunk_id(page.doc_id, [], page.raw_text),
                doc_id=page.doc_id,
                url=page.url,
                title=page.title,
                nav_path=page.nav_path,
                section_path=[],
                chunk_text=page.raw_text,
                code_blocks=page.code_blocks,
                token_estimate=estimate_tokens(page.raw_text),
                fetched_at=page.fetched_at,
            )
        )
    return chunks
