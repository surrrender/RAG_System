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
    assert len(chunks) == 2
    text_chunk = chunks[0]
    code_chunk = chunks[1]
    assert text_chunk.chunk_type == "text"
    assert "短内容" in text_chunk.chunk_text
    assert text_chunk.related_code_ids == [code_chunk.chunk_id]
    assert code_chunk.chunk_type == "code"
    assert "文档路径: 框架 > App > 注册 > 参数" in code_chunk.chunk_text
    assert code_chunk.chunk_text.endswith("code();")
    assert code_chunk.related_text_ids == [text_chunk.chunk_id]


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
    assert len(chunks) == 2
    assert chunks[0].chunk_type == "text"
    assert "文档路径: 框架 > Page" in chunks[0].chunk_text
    assert chunks[0].chunk_text.endswith("page fallback text")
    assert chunks[1].chunk_type == "code"
    assert "文档路径: 框架 > Page" in chunks[1].chunk_text
    assert chunks[1].chunk_text.endswith("Page({})")
    assert chunks[0].related_code_ids == [chunks[1].chunk_id]
    assert chunks[1].related_text_ids == [chunks[0].chunk_id]


def test_build_chunks_creates_separate_code_chunks_per_block() -> None:
    page = PageRecord(
        doc_id="doc-3",
        url="https://example.com/component",
        title="Component",
        nav_path=["框架", "Component"],
        raw_text="full text",
        code_blocks=[],
        source="test",
        fetched_at="2026-03-10T00:00:00+00:00",
        updated_at=None,
    )
    blocks = [
        {
            "section_path": ["注册"],
            "text": "用于注册组件。",
            "code_blocks": ["Component({})", "Component.methods = {}"],
        }
    ]

    chunks = build_chunks(page, blocks, min_chars=20)

    assert [chunk.chunk_type for chunk in chunks] == ["text", "code", "code"]
    assert chunks[0].related_code_ids == [chunks[1].chunk_id, chunks[2].chunk_id]
    assert chunks[1].related_text_ids == [chunks[0].chunk_id]
    assert chunks[2].related_text_ids == [chunks[0].chunk_id]
