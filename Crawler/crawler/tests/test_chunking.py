from crawler.chunking import build_chunks
from crawler.models import PageRecord


def test_build_chunks_keeps_short_sections_separate() -> None:
    page = PageRecord(
        doc_id="doc-1",
        url="https://example.com/app",
        title="App",
        nav_path=["框架", "App"],
        raw_text="full text",
        source="test",
        fetched_at="2026-03-10T00:00:00+00:00",
        updated_at=None,
    )
    blocks = [
        {"section_path": ["注册"], "text": "这是一个足够长的章节内容，超过原先合并阈值。" * 3},
        {"section_path": ["注册", "参数"], "text": "短内容，但不少于十个字不会被丢弃"},
    ]

    chunks = build_chunks(page, blocks, max_chars=500)

    assert len(chunks) == 2
    assert chunks[0].section_path == ["注册"]
    assert chunks[1].section_path == ["注册", "参数"]
    assert chunks[1].chunk_text == "短内容，但不少于十个字不会被丢弃"


def test_build_chunks_splits_oversized_sections_by_sentence() -> None:
    page = PageRecord(
        doc_id="doc-2",
        url="https://example.com/page",
        title="Page",
        nav_path=["框架", "Page"],
        raw_text="page fallback text",
        source="test",
        fetched_at="2026-03-10T00:00:00+00:00",
        updated_at=None,
    )
    sentence_1 = "第一部分介绍小程序注册流程以及基础概念，帮助快速建立整体认知。"
    sentence_2 = "第二部分说明配置项和参数要求，便于排查常见的接入问题。"
    sentence_3 = "第三部分提供调用示例与注意事项，方便开发时直接参考。"
    blocks = [
        {"section_path": ["用法"], "text": f"{sentence_1}{sentence_2}{sentence_3}"},
    ]

    chunks = build_chunks(page, blocks, max_chars=35)

    assert [chunk.chunk_text for chunk in chunks] == [sentence_1, sentence_2, sentence_3]
    assert all(len(chunk.chunk_text) <= 35 for chunk in chunks)


def test_build_chunks_falls_back_to_page_text_and_splits_when_needed() -> None:
    page = PageRecord(
        doc_id="doc-3",
        url="https://example.com/component",
        title="Component",
        nav_path=["框架", "Component"],
        raw_text=(
            "第一段说明整体能力与接入前提，帮助开发者先理解背景。"
            "第二段补充调用方式与限制条件，避免集成时遗漏关键细节。"
        ),
        source="test",
        fetched_at="2026-03-10T00:00:00+00:00",
        updated_at=None,
    )

    chunks = build_chunks(page, [], max_chars=35)

    assert len(chunks) == 2
    assert chunks[0].section_path == []
    assert chunks[1].section_path == []
    assert all(len(chunk.chunk_text) <= 35 for chunk in chunks)


def test_build_chunks_skips_short_text() -> None:
    page = PageRecord(
        doc_id="doc-4",
        url="https://example.com/component",
        title="Component",
        nav_path=["框架", "Component"],
        raw_text="full text",
        source="test",
        fetched_at="2026-03-10T00:00:00+00:00",
        updated_at=None,
    )
    blocks = [
        {"section_path": ["注册"], "text": "用于注册组件。"},
        {"section_path": ["用法"], "text": "用于注册组件。用于注册组件。用于注册组件。"},
    ]

    chunks = build_chunks(page, blocks, max_chars=500)

    assert len(chunks) == 1
