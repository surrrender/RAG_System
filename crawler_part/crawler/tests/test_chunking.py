from crawler.chunking import build_chunks
from crawler.models import PageRecord


def test_build_chunks_merges_short_sections() -> None:
    page = PageRecord(
        doc_id="doc-1",
        url="https://example.com/app",
        title="App",
        nav_path=["框架", "App"],
        raw_text="full text",
        code_blocks=[],
        source="test",
        fetched_at="2026-03-10T00:00:00+00:00",
        updated_at=None,
    )
    blocks = [
        {"section_path": ["注册"], "text": "这是一个足够长的章节内容。" * 10, "code_blocks": []},
        {"section_path": ["注册", "参数"], "text": "短内容", "code_blocks": ["code();"]},
    ]
    chunks = build_chunks(page, blocks, min_chars=20)
    assert len(chunks) == 1
    assert "短内容" in chunks[0].chunk_text
    assert chunks[0].code_blocks == ["code();"]


def test_build_chunks_falls_back_to_page_text() -> None:
    page = PageRecord(
        doc_id="doc-2",
        url="https://example.com/page",
        title="Page",
        nav_path=["框架", "Page"],
        raw_text="page fallback text",
        code_blocks=["Page({})"],
        source="test",
        fetched_at="2026-03-10T00:00:00+00:00",
        updated_at=None,
    )
    chunks = build_chunks(page, [])
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "page fallback text"
