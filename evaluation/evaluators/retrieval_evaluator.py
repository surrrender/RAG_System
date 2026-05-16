from __future__ import annotations

import time
from dataclasses import dataclass, field

from tqdm import tqdm

from evaluation.evaluators.metrics import recall_at_k, mrr, hit_rate
from evaluation.generators.question_generator import TestQuestion


@dataclass(slots=True)
class PerQuestionRetrievalResult:
    question_id: str
    question: str
    category: str
    difficulty: str
    relevant_chunk_ids: set[str]
    retrieved_chunk_ids: list[str]
    retrieval_scores: list[float]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    embed_ms: float = 0.0
    vector_search_ms: float = 0.0
    rerank_ms: float = 0.0


@dataclass(slots=True)
class RetrievalEvalResult:
    questions: list[PerQuestionRetrievalResult]
    avg_recall_at_1: float
    avg_recall_at_3: float
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_mrr: float
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    avg_embed_ms: float
    avg_vector_search_ms: float
    avg_rerank_ms: float

    recall_by_difficulty: dict[str, dict[int, float]] = field(default_factory=dict)
    recall_by_category: dict[str, dict[int, float]] = field(default_factory=dict)
    mrr_by_difficulty: dict[str, float] = field(default_factory=dict)
    mrr_by_category: dict[str, float] = field(default_factory=dict)


# 检索评估
def evaluate_retrieval(
    retriever,
    questions: list[TestQuestion],
    eval_top_k: int = 10,
) -> RetrievalEvalResult:
    results: list[PerQuestionRetrievalResult] = []

    for q in tqdm(questions, desc="Retrieval evaluation"):
        # 这里虽然逻辑上合理,相关的chunk_id应该是个set,但是实际数据集上好像都只有一个相关chunk
        relevant = set(q.relevant_chunk_ids)
        try:
            chunks, metrics = retriever.retrieve_with_metrics(q.question, top_k=eval_top_k)
        except Exception:
            chunks = []
            metrics = type("M", (), {"embed_ms": 0.0, "vector_search_ms": 0.0, "rerank_ms": 0.0})()

        retrieved_ids = [c.chunk_id for c in chunks]
        scores = [c.score for c in chunks]

        embed_ms = float(getattr(metrics, "embed_ms", 0) or 0)
        vec_ms = float(getattr(metrics, "vector_search_ms", 0) or 0)
        rrk_ms = float(getattr(metrics, "rerank_ms", 0) or 0)

        results.append(
            PerQuestionRetrievalResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                difficulty=q.difficulty,
                relevant_chunk_ids=relevant,
                retrieved_chunk_ids=retrieved_ids[:eval_top_k],
                retrieval_scores=scores[:eval_top_k],
                recall_at_1=recall_at_k(relevant, retrieved_ids, 1),
                recall_at_3=recall_at_k(relevant, retrieved_ids, 3),
                recall_at_5=recall_at_k(relevant, retrieved_ids, 5),
                recall_at_10=recall_at_k(relevant, retrieved_ids, 10),
                mrr=mrr(relevant, retrieved_ids),
                hit_at_1=hit_rate(relevant, retrieved_ids, 1),
                hit_at_3=hit_rate(relevant, retrieved_ids, 3),
                hit_at_5=hit_rate(relevant, retrieved_ids, 5),
                hit_at_10=hit_rate(relevant, retrieved_ids, 10),
                embed_ms=embed_ms,
                vector_search_ms=vec_ms,
                rerank_ms=rrk_ms,
            )
        )

    n = max(len(results), 1)
    summary = RetrievalEvalResult(
        questions=results,
        avg_recall_at_1=sum(r.recall_at_1 for r in results) / n,
        avg_recall_at_3=sum(r.recall_at_3 for r in results) / n,
        avg_recall_at_5=sum(r.recall_at_5 for r in results) / n,
        avg_recall_at_10=sum(r.recall_at_10 for r in results) / n,
        avg_mrr=sum(r.mrr for r in results) / n,
        hit_rate_at_1=sum(1 for r in results if r.hit_at_1) / n,
        hit_rate_at_3=sum(1 for r in results if r.hit_at_3) / n,
        hit_rate_at_5=sum(1 for r in results if r.hit_at_5) / n,
        hit_rate_at_10=sum(1 for r in results if r.hit_at_10) / n,
        avg_embed_ms=sum(r.embed_ms for r in results) / n,
        avg_vector_search_ms=sum(r.vector_search_ms for r in results) / n,
        avg_rerank_ms=sum(r.rerank_ms for r in results) / n,
    )

    _compute_groups(results, summary)
    return summary


def _compute_groups(results: list[PerQuestionRetrievalResult], summary: RetrievalEvalResult) -> None:
    for key_attr, store in [("difficulty", summary.recall_by_difficulty), ("category", summary.recall_by_category)]:
        groups: dict[str, list[PerQuestionRetrievalResult]] = {}
        for r in results:
            key = getattr(r, key_attr)
            groups.setdefault(key, []).append(r)
        for key, group_results in groups.items():
            n = max(len(group_results), 1)
            store[key] = {
                1: sum(r.recall_at_1 for r in group_results) / n,
                3: sum(r.recall_at_3 for r in group_results) / n,
                5: sum(r.recall_at_5 for r in group_results) / n,
                10: sum(r.recall_at_10 for r in group_results) / n,
            }

    for key_attr, store in [("difficulty", summary.mrr_by_difficulty), ("category", summary.mrr_by_category)]:
        groups: dict[str, list[PerQuestionRetrievalResult]] = {}
        for r in results:
            key = getattr(r, key_attr)
            groups.setdefault(key, []).append(r)
        for key, group_results in groups.items():
            n = max(len(group_results), 1)
            store[key] = sum(r.mrr for r in group_results) / n
