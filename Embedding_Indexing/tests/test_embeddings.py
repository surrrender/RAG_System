from embedding_indexing.embeddings import chunk_to_embedding_text
from embedding_indexing.models import ChunkRecord


def test_chunk_to_embedding_text_includes_title_path_and_body() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        url="https://example.com/app",
        title="App",
        nav_path=["框架", "App"],
        section_path=["生命周期", "onLaunch"],
        chunk_type="text",
        chunk_text="App 会在小程序初始化时触发 onLaunch。",
        related_code_ids=[],
        related_text_ids=[],
        token_estimate=12,
        fetched_at="2026-03-12T00:00:00+00:00",
    )

    text = chunk_to_embedding_text(chunk)

    assert text == "\n".join(
        [
            "App",
            "框架 > App > 生命周期 > onLaunch",
            "App 会在小程序初始化时触发 onLaunch。",
        ]
    )
