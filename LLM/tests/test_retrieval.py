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

    def fake_build_default_embedder(provider: str, model_name: str, offline: bool, **_: object) -> str:
        assert provider == "sentence-transformer"
        assert model_name == "BAAI/bge-m3"
        assert offline is True
        return "EMBEDDER"

    def fake_build_default_reranker(provider: str, model_name: str, offline: bool, **_: object) -> str:
        assert provider == "cross-encoder"
        assert model_name == "BAAI/bge-reranker-base"
        assert offline is True
        return "RERANKER"

    def fake_search_chunks(**kwargs: object) -> list[dict[str, object]]:
        assert kwargs["embedder"] == "EMBEDDER"
        assert kwargs["reranker"] == "RERANKER"
        assert kwargs["qdrant_url"] is None
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
        return_value=(fake_build_default_embedder, fake_build_default_reranker, fake_search_chunks),
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

    def fake_build_default_embedder(**_: object) -> str:
        return "EMBEDDER"

    def fail_build_default_reranker(**_: object) -> str:
        raise AssertionError("reranker should not be built when disabled")

    def fake_search_chunks(**kwargs: object) -> list[dict[str, object]]:
        assert kwargs["enable_reranker"] is False
        assert kwargs["reranker"] is None
        return []

    with patch(
        "llm.retrieval.load_embedding_indexing_symbols",
        return_value=(fake_build_default_embedder, fail_build_default_reranker, fake_search_chunks),
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
    calls = {"embedder": 0, "reranker": 0}

    def fake_build_default_embedder(**_: object) -> str:
        calls["embedder"] += 1
        return "EMBEDDER"

    def fake_build_default_reranker(**_: object) -> str:
        calls["reranker"] += 1
        return "RERANKER"

    def fake_search_chunks(**_: object) -> list[dict[str, object]]:
        return []

    with patch(
        "llm.retrieval.load_embedding_indexing_symbols",
        return_value=(fake_build_default_embedder, fake_build_default_reranker, fake_search_chunks),
    ):
        retriever.retrieve("first", top_k=2)
        retriever.retrieve("second", top_k=2)

    assert calls == {"embedder": 1, "reranker": 1}
