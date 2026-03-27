from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from embedding_indexing.embeddings import HashEmbedder
from embedding_indexing.pipeline import index_chunks, search_chunks


pytest.importorskip("qdrant_client")


class FakeReranker:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        del query
        scores: list[float] = []
        for document in documents:
            if "App({" in document:
                scores.append(10.0)
            elif "app launch" in document:
                scores.append(5.0)
            else:
                scores.append(1.0)
        return scores


def _write_chunks(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","url":"https://example.com/app","title":"App 生命周期",'
                '"nav_path":["框架"],"section_path":["App"],"chunk_type":"text",'
                '"chunk_text":"app launch show hide lifecycle","related_code_ids":["c3"],'
                '"related_text_ids":[],"token_estimate":5,"fetched_at":"2026-03-12T00:00:00+00:00"}',
                '{"chunk_id":"c2","doc_id":"d2","url":"https://example.com/page","title":"Page 注册",'
                '"nav_path":["框架"],"section_path":["Page"],"chunk_type":"text",'
                '"chunk_text":"page register onload onshow methods","related_code_ids":[],"related_text_ids":[],'
                '"token_estimate":5,"fetched_at":"2026-03-12T00:00:00+00:00"}',
                '{"chunk_id":"c3","doc_id":"d1","url":"https://example.com/app","title":"App 生命周期",'
                '"nav_path":["框架"],"section_path":["App"],"chunk_type":"code",'
                '"chunk_text":"App({ onLaunch() {} })","related_code_ids":[],"related_text_ids":["c1"],'
                '"token_estimate":5,"fetched_at":"2026-03-12T00:00:00+00:00"}',
            ]
        ),
        encoding="utf-8",
    )


def test_index_and_search_round_trip() -> None:
    state_root = Path("state/test_tmp")
    state_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=state_root) as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "chunks.jsonl"
        qdrant_path = temp_path / "qdrant"
        _write_chunks(input_path)

        stats = index_chunks(
            input_path=input_path,
            qdrant_path=qdrant_path,
            collection_name="chunks",
            embedder=HashEmbedder(dimension=48),
            recreate=True,
        )

        results = search_chunks(
            qdrant_path=qdrant_path,
            collection_name="chunks",
            embedder=HashEmbedder(dimension=48),
            query="app launch lifecycle",
            limit=1,
            reranker=FakeReranker(),
            enable_reranker=True,
            rerank_candidate_limit=2,
        )

    assert stats.chunk_count == 3
    assert [result["chunk_type"] for result in results] == ["code"]
    assert results[0]["chunk_id"] == "c3"
    assert results[0]["score"] == pytest.approx(10.0)


def test_search_can_disable_reranker() -> None:
    state_root = Path("state/test_tmp")
    state_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=state_root) as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "chunks.jsonl"
        qdrant_path = temp_path / "qdrant"
        _write_chunks(input_path)

        index_chunks(
            input_path=input_path,
            qdrant_path=qdrant_path,
            collection_name="chunks",
            embedder=HashEmbedder(dimension=48),
            recreate=True,
        )

        results = search_chunks(
            qdrant_path=qdrant_path,
            collection_name="chunks",
            embedder=HashEmbedder(dimension=48),
            query="app launch lifecycle",
            limit=2,
            enable_reranker=False,
            rerank_candidate_limit=1,
        )

    assert [result["chunk_id"] for result in results] == ["c1", "c3"]


def test_search_raises_when_reranker_enabled_without_instance() -> None:
    with pytest.raises(ValueError, match="Reranker is enabled"):
        search_chunks(
            qdrant_path=Path("unused"),
            collection_name="chunks",
            embedder=HashEmbedder(dimension=8),
            query="app launch lifecycle",
            limit=1,
            reranker=None,
            enable_reranker=True,
        )
