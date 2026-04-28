from pathlib import Path
from unittest.mock import patch

from llm.retrieval import Retriever


def test_retriever_normalizes_results() -> None:
    retriever = Retriever(
        qdrant_path=Path("/tmp/qdrant"),
        qdrant_url=None,
        qdrant_api_key=None,
        collection_name="demo",
        embedder_provider="sentence-transformer",
        embedding_model="BAAI/bge-m3",
        reranker_provider="cross-encoder",
        reranker_model="BAAI/bge-reranker-base",
    )

    class FakeEmbedder:
        dimension = 3

    def fake_build_default_embedder(provider: str, model_name: str, offline: bool, **kwargs: object) -> FakeEmbedder:
        assert provider == "sentence-transformer"
        assert model_name == "BAAI/bge-m3"
        assert offline is True
        assert kwargs["device"] == "cpu"
        return FakeEmbedder()

    def fake_build_default_reranker(provider: str, model_name: str, offline: bool, **kwargs: object) -> str:
        assert provider == "cross-encoder"
        assert model_name == "BAAI/bge-reranker-base"
        assert offline is True
        assert kwargs["device"] == "cpu"
        return "RERANKER"

    def fake_initialize_chunk_index(**kwargs: object) -> str:
        assert kwargs["path"] == Path("/tmp/qdrant")
        assert kwargs["collection_name"] == "demo"
        assert kwargs["vector_size"] == 3
        assert kwargs["recreate"] is False
        return "INDEX"

    def fake_search_chunks(**kwargs: object) -> list[dict[str, object]]:
        assert isinstance(kwargs["embedder"], FakeEmbedder)
        assert kwargs["index"] == "INDEX"
        assert kwargs["reranker"] == "RERANKER"
        assert kwargs["enable_reranker"] is True
        assert kwargs["rerank_candidate_limit"] == 10
        assert kwargs["query"] == "test"
        assert kwargs["limit"] == 2
        return [
            {
                "chunk_id": "chunk-1",
                "score": 0.8,
                "title": "Title",
                "url": "https://example.com",
                "section_path": ["A", "B"],
                "chunk_text": "content",
            }
        ]

    with patch(
        "llm.retrieval.load_embedding_indexing_symbols",
        return_value=(
            fake_build_default_embedder,
            fake_build_default_reranker,
            fake_initialize_chunk_index,
            fake_search_chunks,
        ),
    ):
        results = retriever.retrieve("test", top_k=2)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].section_path == ["A", "B"]
    assert results[0].text == "content"


def test_retriever_can_disable_reranker() -> None:
    retriever = Retriever(
        qdrant_path=Path("/tmp/qdrant"),
        qdrant_url=None,
        qdrant_api_key=None,
        collection_name="demo",
        embedder_provider="sentence-transformer",
        embedding_model="BAAI/bge-m3",
        reranker_provider="cross-encoder",
        reranker_model="BAAI/bge-reranker-base",
        disable_reranker=True,
    )

    class FakeEmbedder:
        dimension = 4

    def fake_build_default_embedder(**kwargs: object) -> FakeEmbedder:
        assert kwargs["device"] == "cpu"
        return FakeEmbedder()

    def fail_build_default_reranker(**_: object) -> str:
        raise AssertionError("reranker should not be built when disabled")

    def fake_initialize_chunk_index(**kwargs: object) -> str:
        assert kwargs["vector_size"] == 4
        return "INDEX"

    def fake_search_chunks(**kwargs: object) -> list[dict[str, object]]:
        assert kwargs["enable_reranker"] is False
        assert kwargs["reranker"] is None
        assert kwargs["index"] == "INDEX"
        return []

    with patch(
        "llm.retrieval.load_embedding_indexing_symbols",
        return_value=(
            fake_build_default_embedder,
            fail_build_default_reranker,
            fake_initialize_chunk_index,
            fake_search_chunks,
        ),
    ):
        results = retriever.retrieve("test", top_k=2)

    assert results == []


def test_retriever_reuses_models_across_requests() -> None:
    retriever = Retriever(
        qdrant_path=Path("/tmp/qdrant"),
        qdrant_url=None,
        qdrant_api_key=None,
        collection_name="demo",
        embedder_provider="sentence-transformer",
        embedding_model="BAAI/bge-m3",
        reranker_provider="cross-encoder",
        reranker_model="BAAI/bge-reranker-base",
    )
    calls = {"embedder": 0, "reranker": 0, "index": 0}

    class FakeEmbedder:
        dimension = 8

    def fake_build_default_embedder(**_: object) -> FakeEmbedder:
        calls["embedder"] += 1
        return FakeEmbedder()

    def fake_build_default_reranker(**_: object) -> str:
        calls["reranker"] += 1
        return "RERANKER"

    def fake_initialize_chunk_index(**_: object) -> str:
        calls["index"] += 1
        return "INDEX"

    def fake_search_chunks(**kwargs: object) -> list[dict[str, object]]:
        assert kwargs["index"] == "INDEX"
        return []

    with patch(
        "llm.retrieval.load_embedding_indexing_symbols",
        return_value=(
            fake_build_default_embedder,
            fake_build_default_reranker,
            fake_initialize_chunk_index,
            fake_search_chunks,
        ),
    ):
        retriever.retrieve("first", top_k=2)
        retriever.retrieve("second", top_k=2)

    assert calls == {"embedder": 1, "reranker": 1, "index": 1}
