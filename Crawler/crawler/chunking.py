from __future__ import annotations

from crawler.models import ChunkRecord, PageRecord
from crawler.utils import estimate_tokens, make_chunk_id


def build_chunks(page: PageRecord, heading_blocks: list[dict[str, object]], min_chars: int = 80) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for block in heading_blocks:
        text = str(block["text"]).strip()
        codes = [str(item) for item in block["code_blocks"]]
        section_path = [str(item).lstrip('#') for item in block["section_path"]]
        if not text and not codes:
            continue

        text_chunk = _build_text_chunk(page=page, section_path=section_path, text=text) if text else None
        if text_chunk is not None and chunks and len(text) < min_chars and chunks[-1].chunk_type == "text":
            previous = chunks[-1]
            previous.chunk_text = f"{previous.chunk_text}\n\n{text}".strip()
            previous.token_estimate = estimate_tokens(previous.chunk_text)
            text_chunk = previous
        elif text_chunk is not None:
            chunks.append(text_chunk)

        code_chunks = _build_code_chunks(page=page, section_path=section_path, codes=codes, text_chunk=text_chunk)
        if text_chunk is not None:
            text_chunk.related_code_ids.extend(chunk.chunk_id for chunk in code_chunks)
        chunks.extend(code_chunks)

    if not chunks:
        fallback_text_chunk = _build_text_chunk(page=page, section_path=[], text=page.raw_text)
        assert fallback_text_chunk is not None
        fallback_code_chunks = _build_code_chunks(
            page=page,
            section_path=[],
            codes=page.code_blocks,
            text_chunk=fallback_text_chunk,
        )
        fallback_text_chunk.related_code_ids.extend(chunk.chunk_id for chunk in fallback_code_chunks)
        chunks.append(fallback_text_chunk)
        chunks.extend(fallback_code_chunks)
    return chunks

#TODO:这样其实也不完美,因为在第一步就已经将文本和代码完全切分开了,更合适的方案应该是将文本和代码编织在一起
def build_chunks_with_codes_and_text_together(page: PageRecord, heading_blocks: list[dict[str, object]], min_chars: int = 80) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for block in heading_blocks:
        text = str(block["text"]).strip()
        codes = [str(item) for item in block["code_blocks"]]
        section_path = [str(item).lstrip('#') for item in block["section_path"]]
        if not text and not codes:
            continue

        text_chunk = _build_text_chunk(page=page, section_path=section_path, text=text) if text else None
        if text_chunk is not None and chunks and len(text) < min_chars and chunks[-1].chunk_type == "text":
            previous = chunks[-1]
            previous.chunk_text = f"{previous.chunk_text}\n\n{text}".strip()
            previous.token_estimate = estimate_tokens(previous.chunk_text)
            text_chunk = previous
        elif text_chunk is not None:
            chunks.append(text_chunk)

        code_chunks = _build_code_chunks(page=page, section_path=section_path, codes=codes, text_chunk=text_chunk)
        if text_chunk is not None:
            text_chunk.related_code_ids.extend(chunk.chunk_id for chunk in code_chunks)
        chunks.extend(code_chunks)

    if not chunks:
        fallback_text_chunk = _build_text_chunk(page=page, section_path=[], text=page.raw_text)
        assert fallback_text_chunk is not None
        fallback_code_chunks = _build_code_chunks(
            page=page,
            section_path=[],
            codes=page.code_blocks,
            text_chunk=fallback_text_chunk,
        )
        fallback_text_chunk.related_code_ids.extend(chunk.chunk_id for chunk in fallback_code_chunks)
        chunks.append(fallback_text_chunk)
        chunks.extend(fallback_code_chunks)
    return chunks


def _build_text_chunk(page: PageRecord, section_path: list[str], text: str) -> ChunkRecord | None:
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
        chunk_type="text",
        chunk_text=body,
        related_code_ids=[],
        related_text_ids=[],
        token_estimate=estimate_tokens(body),
        fetched_at=page.fetched_at,
    )


def _build_code_chunks(
    page: PageRecord,
    section_path: list[str],
    codes: list[str],
    text_chunk: ChunkRecord | None,
) -> list[ChunkRecord]:
    code_chunks: list[ChunkRecord] = []
    related_text_ids = [text_chunk.chunk_id] if text_chunk is not None else []
    for index, code in enumerate(codes, start=1):
        body = code.strip()

        if not body:
            continue
        code_chunks.append(
            ChunkRecord(
                chunk_id=make_chunk_id(
                    page.doc_id,
                    section_path,
                    body,
                    chunk_type="code",
                    salt=f"code-{index}",
                ),
                doc_id=page.doc_id,
                url=page.url,
                title=page.title,
                nav_path=page.nav_path,
                section_path=section_path,
                chunk_type="code",
                chunk_text=body,
                related_code_ids=[],
                related_text_ids=related_text_ids.copy(),
                token_estimate=estimate_tokens(body),
                fetched_at=page.fetched_at,
            )
        )
    return code_chunks
