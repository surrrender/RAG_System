from __future__ import annotations

import time
import concurrent.futures
from dataclasses import dataclass, field
from collections.abc import Iterator

from tqdm import tqdm


@dataclass(slots=True)
class SingleQueryPerformance:
    question_id: str
    total_ms: float
    embed_ms: float = 0.0
    vector_search_ms: float = 0.0
    rerank_ms: float = 0.0
    prompt_build_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0


@dataclass(slots=True)
class PerformanceResult:
    queries: list[SingleQueryPerformance]

    # Averages
    avg_total_ms: float = 0.0
    avg_embed_ms: float = 0.0
    avg_vector_search_ms: float = 0.0
    avg_rerank_ms: float = 0.0
    avg_generation_ms: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0

    # Percentiles
    p50_total_ms: float = 0.0
    p95_total_ms: float = 0.0
    p99_total_ms: float = 0.0
    throughput_qps: float = 0.0

    # Concurrency
    concurrency_results: list[dict] = field(default_factory=list)


def run_single_query_performance(
    service,
    questions: list,
    top_k: int = 5,
) -> PerformanceResult:
    results: list[SingleQueryPerformance] = []

    for q in tqdm(questions, desc="Performance testing"):
        t0 = time.perf_counter()
        try:
            prepared = service._prepare_answer(q.question, top_k=top_k)
            prep_ms = _elapsed_ms(t0)

            gen_t0 = time.perf_counter()
            answer = service._generator.generate(prepared.prompt)
            gen_ms = _elapsed_ms(gen_t0)

            total_ms = _elapsed_ms(t0)

            results.append(SingleQueryPerformance(
                question_id=q.id,
                total_ms=total_ms,
                embed_ms=prepared.retrieval_metrics.embed_ms or 0,
                vector_search_ms=prepared.retrieval_metrics.vector_search_ms or 0,
                rerank_ms=prepared.retrieval_metrics.rerank_ms or 0,
                prompt_build_ms=prepared.prompt_build_ms or 0,
                generation_ms=gen_ms,
                prompt_chars=len(prepared.prompt),
                completion_chars=len(answer),
                prompt_tokens=_estimate_tokens(prepared.prompt),
                completion_tokens=_estimate_tokens(answer),
            ))
        except Exception:
            results.append(SingleQueryPerformance(
                question_id=q.id,
                total_ms=_elapsed_ms(t0),
            ))

    return _compute_performance_summary(results)


def run_concurrency_performance(
    service,
    questions: list,
    concurrency_levels: list[int] = [1, 2, 4],
    top_k: int = 5,
) -> list[dict]:
    all_concurrency: list[dict] = []

    for concurrency in concurrency_levels:
        test_questions = questions[:max(10, concurrency * 3)]

        latencies: list[float] = []
        errors: int = 0
        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for q in test_questions:
                futures.append(executor.submit(_safe_answer, service, q.question, top_k))
            for f in concurrent.futures.as_completed(futures):
                try:
                    result = f.result()
                    latencies.append(result["latency_ms"])
                    if result.get("error"):
                        errors += 1
                except Exception:
                    errors += 1

        total_time = _elapsed_ms(t0)
        total_queries = len(test_questions)
        qps = total_queries / (total_time / 1000) if total_time > 0 else 0

        all_concurrency.append({
            "concurrency": concurrency,
            "total_queries": total_queries,
            "total_time_ms": total_time,
            "qps": round(qps, 2),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "errors": errors,
        })

    return all_concurrency


def _compute_performance_summary(results: list[SingleQueryPerformance]) -> PerformanceResult:
    n = max(len(results), 1)
    latencies = [r.total_ms for r in results if r.total_ms > 0]

    return PerformanceResult(
        queries=results,
        avg_total_ms=sum(r.total_ms for r in results) / n,
        avg_embed_ms=sum(r.embed_ms for r in results) / n,
        avg_vector_search_ms=sum(r.vector_search_ms for r in results) / n,
        avg_rerank_ms=sum(r.rerank_ms for r in results) / n,
        avg_generation_ms=sum(r.generation_ms for r in results) / n,
        avg_prompt_tokens=sum(r.prompt_tokens for r in results) / n,
        avg_completion_tokens=sum(r.completion_tokens for r in results) / n,
        p50_total_ms=_percentile(latencies, 50),
        p95_total_ms=_percentile(latencies, 95),
        p99_total_ms=_percentile(latencies, 99),
        throughput_qps=1000 / (sum(r.total_ms for r in results) / n) if n > 0 else 0,
    )


def _safe_answer(service, question: str, top_k: int) -> dict:
    t0 = time.perf_counter()
    try:
        prepared = service._prepare_answer(question, top_k=top_k)
        answer = service._generator.generate(prepared.prompt)
        return {"latency_ms": _elapsed_ms(t0), "error": False}
    except Exception:
        return {"latency_ms": _elapsed_ms(t0), "error": True}


def _elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 3)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]


def _estimate_tokens(text: str) -> int:
    cn_chars = sum(1 for c in text if ord(c) > 127)
    en_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + en_chars / 4.0)
