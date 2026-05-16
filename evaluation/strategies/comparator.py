from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from evaluation.evaluators.metrics import recall_at_k, mrr, hit_rate
from evaluation.generators.question_generator import TestQuestion
from evaluation.strategies.hybrid import HybridRetriever, StrategyResult


@dataclass(slots=True)
class PerStrategyPerQuestion:
    question_id: str
    question: str
    category: str
    difficulty: str
    strategy: str
    chunk_ids: list[str]
    scores: list[float]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    hit_at_5: bool
    total_ms: float


@dataclass(slots=True)
class StrategyComparisonResult:
    per_strategy_questions: list[PerStrategyPerQuestion] = field(default_factory=list)

    # Per-strategy summaries
    dense_recall_5: float = 0.0
    dense_recall_10: float = 0.0
    dense_mrr: float = 0.0
    dense_hit_5: float = 0.0
    dense_avg_ms: float = 0.0

    bm25_recall_5: float = 0.0
    bm25_recall_10: float = 0.0
    bm25_mrr: float = 0.0
    bm25_hit_5: float = 0.0
    bm25_avg_ms: float = 0.0

    hybrid_recall_5: float = 0.0
    hybrid_recall_10: float = 0.0
    hybrid_mrr: float = 0.0
    hybrid_hit_5: float = 0.0
    hybrid_avg_ms: float = 0.0

    hybrid_rerank_recall_5: float = 0.0
    hybrid_rerank_recall_10: float = 0.0
    hybrid_rerank_mrr: float = 0.0
    hybrid_rerank_hit_5: float = 0.0
    hybrid_rerank_avg_ms: float = 0.0

    # Grouped by difficulty
    by_difficulty: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    # Improvement percentages
    hybrid_vs_dense_improvement: float = 0.0
    rerank_vs_hybrid_improvement: float = 0.0


class StrategyComparator:
    def __init__(self, hybrid_retriever: HybridRetriever) -> None:
        self._hr = hybrid_retriever

    def compare(self, questions: list[TestQuestion], top_k: int = 10) -> StrategyComparisonResult:
        all_results: list[PerStrategyPerQuestion] = []

        for q in tqdm(questions, desc="Strategy comparison"):
            relevant = set(q.relevant_chunk_ids)

            strategies: list[StrategyResult] = []
            try:
                strategies.append(self._hr.search_dense_only(q.question, top_k=top_k))
            except Exception:
                strategies.append(StrategyResult("Dense", [], [], {"total_ms": 0}))
            try:
                strategies.append(self._hr.search_bm25_only(q.question, top_k=top_k))
            except Exception:
                strategies.append(StrategyResult("BM25", [], [], {"total_ms": 0}))
            try:
                strategies.append(self._hr.search_hybrid(q.question, top_k=top_k))
            except Exception:
                strategies.append(StrategyResult("Hybrid", [], [], {"total_ms": 0}))
            try:
                strategies.append(self._hr.search_hybrid_rerank(q.question, top_k=top_k))
            except Exception:
                strategies.append(StrategyResult("Hybrid+Reranker", [], [], {"total_ms": 0}))

            for sr in strategies:
                all_results.append(PerStrategyPerQuestion(
                    question_id=q.id,
                    question=q.question,
                    category=q.category,
                    difficulty=q.difficulty,
                    strategy=sr.strategy_name,
                    chunk_ids=sr.chunk_ids,
                    scores=sr.scores,
                    recall_at_5=recall_at_k(relevant, sr.chunk_ids, 5),
                    recall_at_10=recall_at_k(relevant, sr.chunk_ids, 10),
                    mrr=mrr(relevant, sr.chunk_ids),
                    hit_at_5=hit_rate(relevant, sr.chunk_ids, 5),
                    total_ms=sr.timing.get("total_ms", 0),
                ))

        result = StrategyComparisonResult(per_strategy_questions=all_results)
        _compute_strategy_summaries(result, all_results)
        return result


def _compute_strategy_summaries(result: StrategyComparisonResult, all_results: list[PerStrategyPerQuestion]) -> None:
    by_strategy: dict[str, list[PerStrategyPerQuestion]] = {}
    for r in all_results:
        by_strategy.setdefault(r.strategy, []).append(r)

    for strategy, items in by_strategy.items():
        n = max(len(items), 1)
        r5 = sum(i.recall_at_5 for i in items) / n
        r10 = sum(i.recall_at_10 for i in items) / n
        mr = sum(i.mrr for i in items) / n
        h5 = sum(1 for i in items if i.hit_at_5) / n
        ms = sum(i.total_ms for i in items) / n

        if strategy == "Dense":
            result.dense_recall_5, result.dense_recall_10 = r5, r10
            result.dense_mrr, result.dense_hit_5 = mr, h5
            result.dense_avg_ms = ms
        elif strategy == "BM25":
            result.bm25_recall_5, result.bm25_recall_10 = r5, r10
            result.bm25_mrr, result.bm25_hit_5 = mr, h5
            result.bm25_avg_ms = ms
        elif strategy == "Hybrid":
            result.hybrid_recall_5, result.hybrid_recall_10 = r5, r10
            result.hybrid_mrr, result.hybrid_hit_5 = mr, h5
            result.hybrid_avg_ms = ms
        elif strategy == "Hybrid+Reranker":
            result.hybrid_rerank_recall_5, result.hybrid_rerank_recall_10 = r5, r10
            result.hybrid_rerank_mrr, result.hybrid_rerank_hit_5 = mr, h5
            result.hybrid_rerank_avg_ms = ms

    # Group by difficulty
    by_diff: dict[str, dict[str, list[PerStrategyPerQuestion]]] = {}
    for r in all_results:
        by_diff.setdefault(r.difficulty, {}).setdefault(r.strategy, []).append(r)

    for diff, strategies in by_diff.items():
        result.by_difficulty.setdefault(diff, {})
        for strategy, items in strategies.items():
            n = max(len(items), 1)
            result.by_difficulty[diff][strategy] = {
                "recall_5": sum(i.recall_at_5 for i in items) / n,
                "recall_10": sum(i.recall_at_10 for i in items) / n,
                "mrr": sum(i.mrr for i in items) / n,
                "hit_5": sum(1 for i in items if i.hit_at_5) / n,
            }

    # Improvement percentages
    if result.dense_recall_5 > 0:
        result.hybrid_vs_dense_improvement = (result.hybrid_recall_5 - result.dense_recall_5) / result.dense_recall_5 * 100
    if result.hybrid_recall_5 > 0:
        result.rerank_vs_hybrid_improvement = (result.hybrid_rerank_recall_5 - result.hybrid_recall_5) / result.hybrid_recall_5 * 100
