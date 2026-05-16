from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.strategies.bm25 import BM25Result, build_bm25_index, BM25Retriever


@dataclass(slots=True)
class HybridResult:
    chunk_id: str
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float | None = None
    title: str | None = None
    url: str | None = None
    section_path: list[str] | None = None
    text: str | None = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        return self.hybrid_score


@dataclass(slots=True)
class StrategyResult:
    strategy_name: str
    chunk_ids: list[str]
    scores: list[float]
    timing: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        embedder,
        qdrant_index,
        bm25: BM25Retriever,
        reranker=None,
        rerank_candidate_limit: int = 15,
        dense_weight: float = 0.5,
        k_rrf: int = 60,
    ) -> None:
        self._embedder = embedder
        self._index = qdrant_index
        self._bm25 = bm25
        self._reranker = reranker
        self._rerank_candidate_limit = rerank_candidate_limit
        self._dense_weight = dense_weight
        self._k_rrf = k_rrf

    def search_dense_only(self, query: str, top_k: int = 5) -> StrategyResult:
        t0 = time.perf_counter()
        from embedding_indexing.pipeline import search_chunks

        stage_metrics: dict[str, float] = {}
        results = search_chunks(
            index=self._index, embedder=self._embedder, query=query,
            limit=top_k, reranker=None, enable_reranker=False,
            rerank_candidate_limit=10, stage_metrics=stage_metrics,
        )
        return StrategyResult(
            strategy_name="Dense",
            chunk_ids=[str(r["chunk_id"]) for r in results],
            scores=[float(r["score"]) for r in results],
            timing={
                "total_ms": _elapsed_ms(t0),
                "embed_ms": stage_metrics.get("embed_ms", 0),
                "vector_search_ms": stage_metrics.get("vector_search_ms", 0),
            },
        )

    def search_bm25_only(self, query: str, top_k: int = 5) -> StrategyResult:
        t0 = time.perf_counter()
        results = self._bm25.search(query, top_k=top_k)
        return StrategyResult(
            strategy_name="BM25",
            chunk_ids=[r.chunk_id for r in results],
            scores=[r.score for r in results],
            timing={"total_ms": _elapsed_ms(t0)},
        )

    def search_hybrid(self, query: str, top_k: int = 5) -> StrategyResult:
        t0 = time.perf_counter()

        # Dense search with larger candidate pool
        from embedding_indexing.pipeline import search_chunks

        stage_metrics: dict[str, float] = {}
        candidate_limit = max(top_k, 20)
        dense_results = search_chunks(
            index=self._index, embedder=self._embedder, query=query,
            limit=candidate_limit, reranker=None, enable_reranker=False,
            rerank_candidate_limit=10, stage_metrics=stage_metrics,
        )

        # BM25 search with larger candidate pool
        bm25_results = self._bm25.search(query, top_k=candidate_limit)

        # Merge with RRF
        merged = self._rrf_merge(dense_results, bm25_results, top_k)

        return StrategyResult(
            strategy_name="Hybrid",
            chunk_ids=[m.chunk_id for m in merged],
            scores=[m.hybrid_score for m in merged],
            timing={
                "total_ms": _elapsed_ms(t0),
                "embed_ms": stage_metrics.get("embed_ms", 0),
                "vector_search_ms": stage_metrics.get("vector_search_ms", 0),
                "bm25_ms": 0,
            },
        )

    def search_hybrid_rerank(self, query: str, top_k: int = 5) -> StrategyResult:
        t0 = time.perf_counter()

        from embedding_indexing.pipeline import search_chunks

        stage_metrics: dict[str, float] = {}
        candidate_limit = max(top_k, self._rerank_candidate_limit)
        dense_results = search_chunks(
            index=self._index, embedder=self._embedder, query=query,
            limit=candidate_limit, reranker=None, enable_reranker=False,
            rerank_candidate_limit=10, stage_metrics=stage_metrics,
        )
        bm25_results = self._bm25.search(query, top_k=candidate_limit)

        merged = self._rrf_merge(dense_results, bm25_results, candidate_limit)

        # Rerank the merged candidates
        if self._reranker is not None and len(merged) > top_k:
            rr_t0 = time.perf_counter()
            documents = [m.text or "" for m in merged]
            rerank_scores = self._reranker.rerank(query, documents)
            for i, m in enumerate(merged):
                m.rerank_score = float(rerank_scores[i]) if i < len(rerank_scores) else 0.0
            merged.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            rerank_ms = _elapsed_ms(rr_t0)
        else:
            rerank_ms = 0

        top = merged[:top_k]

        return StrategyResult(
            strategy_name="Hybrid+Reranker",
            chunk_ids=[m.chunk_id for m in top],
            scores=[m.final_score for m in top],
            timing={
                "total_ms": _elapsed_ms(t0),
                "embed_ms": stage_metrics.get("embed_ms", 0),
                "vector_search_ms": stage_metrics.get("vector_search_ms", 0),
                "rerank_ms": rerank_ms,
            },
        )

    def _rrf_merge(
        self,
        dense_results: list[dict],
        bm25_results: list[BM25Result],
        top_k: int,
    ) -> list[HybridResult]:
        rrf_map: dict[str, dict] = {}

        for rank, r in enumerate(dense_results, start=1):
            cid = str(r["chunk_id"])
            rrf_map[cid] = {
                "chunk_id": cid,
                "dense_score": float(r["score"]),
                "bm25_score": 0.0,
                "title": r.get("title"),
                "url": r.get("url"),
                "section_path": r.get("section_path"),
                "text": r.get("chunk_text"),
                "dense_rrf": 1.0 / (self._k_rrf + rank),
                "bm25_rrf": 0.0,
            }

        for rank, r in enumerate(bm25_results, start=1):
            cid = r.chunk_id
            if cid in rrf_map:
                rrf_map[cid]["bm25_rrf"] = 1.0 / (self._k_rrf + rank)
                rrf_map[cid]["bm25_score"] = r.score
            else:
                rrf_map[cid] = {
                    "chunk_id": cid,
                    "dense_score": 0.0,
                    "bm25_score": r.score,
                    "title": r.title,
                    "url": r.url,
                    "section_path": r.section_path,
                    "text": r.text,
                    "dense_rrf": 0.0,
                    "bm25_rrf": 1.0 / (self._k_rrf + rank),
                }

        results: list[dict] = []
        for cid, data in rrf_map.items():
            data["hybrid_score"] = (
                self._dense_weight * data["dense_rrf"] + (1 - self._dense_weight) * data["bm25_rrf"]
            )
            results.append(data)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return [
            HybridResult(
                chunk_id=r["chunk_id"],
                dense_score=r["dense_score"],
                bm25_score=r["bm25_score"],
                hybrid_score=r["hybrid_score"],
                title=r.get("title"),
                url=r.get("url"),
                section_path=r.get("section_path"),
                text=r.get("text"),
            )
            for r in results[:top_k]
        ]


def _elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 3)
