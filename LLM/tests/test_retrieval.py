from pathlib import Path
from unittest.mock import patch

from llm.retrieval import Retriever


def test_retriever_normalizes_results() -> None:
    retriever = Retriever(
        qdrant_path=Path("/tmp/qdrant"),
        collection_name="demo",
        embedder_provider="sentence-transformer",
        embedding_model="BAAI/bge-m3",
    )

    def fake_build_default_embedder(provider: str, model_name: str, offline: bool, **_: object) -> str:
        assert provider == "sentence-transformer"
        assert model_name == "BAAI/bge-m3"
        assert offline is True
        return "EMBEDDER"

    def fake_search_chunks(**kwargs: object) -> list[dict[str, object]]:
        assert kwargs["embedder"] == "EMBEDDER"
        assert kwargs["query"] == "test"
        assert kwargs["limit"] == 2
        return [
            {
                "chunk_id": "chunk-1",
                "score": 0.8,
                "title": "Title",
                "url": "https://example.com",
                "section_path": ["A", "B"],
                "text": "content",
            }
        ]

    with patch("llm.retrieval.load_embedding_indexing_symbols", return_value=(fake_build_default_embedder, fake_search_chunks)):
        results = retriever.retrieve("test", top_k=2)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].section_path == ["A", "B"]
