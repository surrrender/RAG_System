import json

from crawler.models import ChunkRecord


def test_chunk_record_serialization_fields_complete() -> None:
    record = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        url="https://example.com",
        title="Title",
        nav_path=["框架"],
        section_path=["章节"],
        chunk_type="text",
        chunk_text="body",
        token_estimate=1,
        fetched_at="2026-03-10T00:00:00+00:00",
        related_code_ids=["chunk-2"],
        related_text_ids=[],
    )
    serialized = json.loads(json.dumps(record.to_dict(), ensure_ascii=False))
    assert set(serialized) == {
        "chunk_id",
        "doc_id",
        "url",
        "title",
        "nav_path",
        "section_path",
        "chunk_type",
        "chunk_text",
        "token_estimate",
        "fetched_at",
        "related_code_ids",
        "related_text_ids",
    }
