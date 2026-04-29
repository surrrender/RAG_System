from embedding_indexing.pipeline import search_chunks


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        assert text == "app lifecycle"
        return [0.1, 0.2]


class FakePoint:
    def __init__(self, chunk_id: str, score: float, text: str) -> None:
        self.id = chunk_id
        self.score = score
        self.payload = {
            "chunk_id": chunk_id,
            "title": "App",
            "url": "https://example.com/app",
            "section_path": ["生命周期"],
            "chunk_type": "text",
            "text": text,
        }


class FakeIndex:
    def __init__(self, points: list[FakePoint]) -> None:
        self._points = points
        self.calls: list[tuple[list[float], int]] = []

    def search(self, query_vector: list[float], limit: int = 5) -> list[FakePoint]:
        self.calls.append((query_vector, limit))
        return self._points[:limit]


class FailIfCalledReranker:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise AssertionError("rerank should be skipped when candidate count does not exceed top_k")


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return [2.0, 1.0, 3.0]


def test_search_chunks_skips_rerank_when_candidate_count_does_not_exceed_limit() -> None:
    stage_metrics: dict[str, float] = {}
    index = FakeIndex(
        [
            FakePoint("chunk-1", 0.9, "doc-1"),
            FakePoint("chunk-2", 0.8, "doc-2"),
        ]
    )

    results = search_chunks(
        index=index,
        embedder=FakeEmbedder(),
        query="app lifecycle",
        limit=2,
        reranker=FailIfCalledReranker(),
        enable_reranker=True,
        rerank_candidate_limit=5,
        stage_metrics=stage_metrics,
    )

    assert [item["chunk_id"] for item in results] == ["chunk-1", "chunk-2"]
    assert "embed_ms" in stage_metrics
    assert "vector_search_ms" in stage_metrics
    assert "rerank_ms" not in stage_metrics


def test_search_chunks_records_rerank_timing_when_rerank_runs() -> None:
    stage_metrics: dict[str, float] = {}
    reranker = RecordingReranker()
    index = FakeIndex(
        [
            FakePoint("chunk-1", 0.9, "doc-1"),
            FakePoint("chunk-2", 0.8, "doc-2"),
            FakePoint("chunk-3", 0.7, "doc-3"),
        ]
    )

    results = search_chunks(
        index=index,
        embedder=FakeEmbedder(),
        query="app lifecycle",
        limit=2,
        reranker=reranker,
        enable_reranker=True,
        rerank_candidate_limit=3,
        stage_metrics=stage_metrics,
    )

    assert [item["chunk_id"] for item in results] == ["chunk-3", "chunk-1"]
    assert reranker.calls == [("app lifecycle", ["doc-1", "doc-2", "doc-3"])]
    assert "embed_ms" in stage_metrics
    assert "vector_search_ms" in stage_metrics
    assert "rerank_ms" in stage_metrics
