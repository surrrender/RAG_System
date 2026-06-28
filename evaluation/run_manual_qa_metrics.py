"""
Manual QA evaluation runner for thesis-ready metrics.

This script:
1. Loads questions from evaluation/data/manual_questions.json
2. Calls the local LLM backend (default: http://127.0.0.1:8000) to generate answers
3. Caches generated answers to avoid repeated Ollama calls
4. Computes multiple automatic metrics
5. Saves per-question results and aggregate summaries for paper use

Usage:
    source .venv/bin/activate
    python -m evaluation.run_manual_qa_metrics

Optional:
    python -m evaluation.run_manual_qa_metrics --limit 5
    python -m evaluation.run_manual_qa_metrics --skip-generation
    python -m evaluation.run_manual_qa_metrics --force-regenerate
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "evaluation" / "data" / "manual_questions.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "outputs" / "manual_qa_metrics"
DEFAULT_ANSWERS_PATH = DEFAULT_OUTPUT_DIR / "generated_answers.json"
DEFAULT_RESULTS_JSON_PATH = DEFAULT_OUTPUT_DIR / "per_question_metrics.json"
DEFAULT_RESULTS_CSV_PATH = DEFAULT_OUTPUT_DIR / "per_question_metrics.csv"
DEFAULT_SUMMARY_JSON_PATH = DEFAULT_OUTPUT_DIR / "summary.json"
DEFAULT_SUMMARY_MD_PATH = DEFAULT_OUTPUT_DIR / "summary.md"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TOP_K = 5
DEFAULT_PROVIDER = "ollama"
DEFAULT_TIMEOUT = 300.0
DEFAULT_EVAL_USER_ID = "manual-qa-eval"
BGE_SMALL_ZH_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--BAAI--bge-small-zh-v1.5"
    / "snapshots"
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
MARKDOWN_BULLET_PATTERN = re.compile(r"^\s*([-*+]|\d+[.)])\s+", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SMOOTHING = SmoothingFunction().method1
CHRF_SCORER = CHRF(word_order=0)
ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
)
METRIC_KEYS = [
    "bleu_1",
    "bleu_2",
    "bleu_4",
    "rouge_1_f",
    "rouge_2_f",
    "rouge_l_f",
    "meteor",
    "bertscore_f1",
    "chrf",
    "exact_match",
    "f1",
]


@dataclass(slots=True)
class QuestionItem:
    id: str
    question: str
    reference_answer: str
    category: str
    difficulty: str


class EmptyWordNet:
    def synsets(self, _: str) -> list[Any]:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual QA evaluation with cached answers.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--user-id", default=DEFAULT_EVAL_USER_ID)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--bertscore-model", default=None)
    parser.add_argument("--bertscore-num-layers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_questions(args.dataset, limit=args.limit)
    answers_path = output_dir / DEFAULT_ANSWERS_PATH.name
    results_json_path = output_dir / DEFAULT_RESULTS_JSON_PATH.name
    results_csv_path = output_dir / DEFAULT_RESULTS_CSV_PATH.name
    summary_json_path = output_dir / DEFAULT_SUMMARY_JSON_PATH.name
    summary_md_path = output_dir / DEFAULT_SUMMARY_MD_PATH.name

    cached_answers = load_cached_answers(answers_path)
    answers = generate_or_reuse_answers(
        questions=dataset,
        cached_answers=cached_answers,
        answers_path=answers_path,
        base_url=args.base_url.rstrip("/"),
        provider=args.provider,
        top_k=args.top_k,
        timeout=args.timeout,
        user_id=args.user_id,
        skip_generation=args.skip_generation,
        force_regenerate=args.force_regenerate,
    )

    bertscore_model_ref = resolve_bertscore_model(args.bertscore_model)
    bertscore_num_layers = resolve_bertscore_num_layers(
        bertscore_model_ref,
        explicit=args.bertscore_num_layers,
    )

    metrics_payload = compute_metrics_payload(
        questions=dataset,
        answers=answers,
        bertscore_model_ref=bertscore_model_ref,
        bertscore_num_layers=bertscore_num_layers,
    )
    summary_payload = build_summary_payload(
        metrics_payload=metrics_payload,
        dataset_path=args.dataset,
        answers_path=answers_path,
        base_url=args.base_url.rstrip("/"),
        provider=args.provider,
        top_k=args.top_k,
        bertscore_model_ref=bertscore_model_ref,
        bertscore_num_layers=bertscore_num_layers,
    )

    results_json_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_metrics_csv(metrics_payload["results"], results_csv_path)
    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md_path.write_text(render_summary_markdown(summary_payload), encoding="utf-8")

    print("=" * 72)
    print("Manual QA evaluation finished")
    print("=" * 72)
    print(f"Answers cache   : {answers_path}")
    print(f"Per-question JSON: {results_json_path}")
    print(f"Per-question CSV : {results_csv_path}")
    print(f"Summary JSON     : {summary_json_path}")
    print(f"Summary Markdown : {summary_md_path}")
    print()
    print("Average metrics (all questions, higher is better):")
    for item in summary_payload["aggregate"]["ranking_by_mean"]:
        print(f"  {item['metric']:<14} {item['mean']:.4f}")


def load_questions(dataset_path: Path, limit: int | None = None) -> list[QuestionItem]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = [
        QuestionItem(
            id=item["id"],
            question=item["question"],
            reference_answer=item["reference_answer"],
            category=item["category"],
            difficulty=item["difficulty"],
        )
        for item in payload["questions"]
    ]
    return items[:limit] if limit else items


def load_cached_answers(answers_path: Path) -> dict[str, dict[str, Any]]:
    if not answers_path.exists():
        return {}
    payload = json.loads(answers_path.read_text(encoding="utf-8"))
    answers = payload.get("answers", [])
    return {item["id"]: item for item in answers}


def generate_or_reuse_answers(
    *,
    questions: list[QuestionItem],
    cached_answers: dict[str, dict[str, Any]],
    answers_path: Path,
    base_url: str,
    provider: str,
    top_k: int,
    timeout: float,
    user_id: str,
    skip_generation: bool,
    force_regenerate: bool,
) -> list[dict[str, Any]]:
    answers_by_id = dict(cached_answers)
    progress = tqdm(questions, desc="Generating answers")
    for question in progress:
        cached = answers_by_id.get(question.id)
        if should_reuse_cached_answer(
            cached=cached,
            question=question,
            skip_generation=skip_generation,
            force_regenerate=force_regenerate,
            base_url=base_url,
            provider=provider,
            top_k=top_k,
        ):
            progress.set_postfix_str(f"{question.id} cache")
            continue

        if skip_generation and cached is None:
            raise RuntimeError(
                f"--skip-generation was set, but no cached answer exists for {question.id}."
            )

        progress.set_postfix_str(question.id)
        started_at = time.perf_counter()
        record: dict[str, Any]
        try:
            answer_result = request_answer(
                base_url=base_url,
                user_id=user_id,
                question=question.question,
                top_k=top_k,
                provider=provider,
                timeout=timeout,
            )
            latency_seconds = round(time.perf_counter() - started_at, 3)
            record = {
                "id": question.id,
                "question": question.question,
                "reference_answer": question.reference_answer,
                "category": question.category,
                "difficulty": question.difficulty,
                "generated_answer": answer_result["answer"],
                "status": "success",
                "error": None,
                "model": answer_result.get("model"),
                "retrieval_count": answer_result.get("retrieval_count"),
                "citations": answer_result.get("citations", []),
                "latency_seconds": latency_seconds,
                "base_url": base_url,
                "provider": provider,
                "top_k": top_k,
                "generated_at": utc_now(),
            }
        except Exception as exc:  # pragma: no cover - depends on runtime services
            latency_seconds = round(time.perf_counter() - started_at, 3)
            record = {
                "id": question.id,
                "question": question.question,
                "reference_answer": question.reference_answer,
                "category": question.category,
                "difficulty": question.difficulty,
                "generated_answer": "",
                "status": "error",
                "error": str(exc),
                "model": None,
                "retrieval_count": 0,
                "citations": [],
                "latency_seconds": latency_seconds,
                "base_url": base_url,
                "provider": provider,
                "top_k": top_k,
                "generated_at": utc_now(),
            }
        answers_by_id[question.id] = record
        save_answers_cache(answers_by_id, answers_path)

    ordered_answers = [answers_by_id[question.id] for question in questions]
    save_answers_cache(answers_by_id, answers_path)
    return ordered_answers


def should_reuse_cached_answer(
    *,
    cached: dict[str, Any] | None,
    question: QuestionItem,
    skip_generation: bool,
    force_regenerate: bool,
    base_url: str,
    provider: str,
    top_k: int,
) -> bool:
    if cached is None:
        return False
    if skip_generation:
        return True
    if force_regenerate:
        return False
    return (
        cached.get("question") == question.question
        and cached.get("reference_answer") == question.reference_answer
        and cached.get("base_url") == base_url
        and cached.get("provider") == provider
        and int(cached.get("top_k", -1)) == top_k
        and cached.get("status") == "success"
    )


def save_answers_cache(answers_by_id: dict[str, dict[str, Any]], answers_path: Path) -> None:
    ordered = sorted(answers_by_id.values(), key=lambda item: item["id"])
    payload = {
        "meta": {
            "saved_at": utc_now(),
            "answer_count": len(ordered),
        },
        "answers": ordered,
    }
    answers_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def request_answer(
    *,
    base_url: str,
    user_id: str,
    question: str,
    top_k: int,
    provider: str,
    timeout: float,
) -> dict[str, Any]:
    conversation = request_json(
        "POST",
        f"{base_url}/conversations",
        {"user_id": user_id},
        timeout=timeout,
    )
    conversation_id = conversation["id"]
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "question": question,
        "top_k": top_k,
        "generation_provider": provider,
    }
    try:
        return request_json("POST", f"{base_url}/qa", payload, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    return request_answer_via_stream(base_url=base_url, payload=payload, timeout=timeout)


def request_json(method: str, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_answer_via_stream(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/qa/stream",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        meta: dict[str, Any] = {}
        citations: list[dict[str, Any]] = []
        answer = ""
        for event_name, data in iter_sse_events(response):
            if event_name == "meta":
                meta = data
            elif event_name == "delta":
                answer += str(data.get("text") or "")
            elif event_name == "citations":
                citations = data.get("citations", [])
            elif event_name == "error":
                raise RuntimeError(str(data.get("message") or "stream generation failed"))
            elif event_name == "done":
                answer = str(data.get("answer") or answer)
                break
    return {
        "question": payload["question"],
        "answer": answer,
        "citations": citations,
        "model": meta.get("model"),
        "retrieval_count": meta.get("retrieval_count", 0),
    }


def iter_sse_events(response: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if event_name is not None:
                payload = "\n".join(data_lines) if data_lines else "{}"
                yield event_name, json.loads(payload)
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data_lines.append(line.partition(":")[2].strip())


def compute_metrics_payload(
    *,
    questions: list[QuestionItem],
    answers: list[dict[str, Any]],
    bertscore_model_ref: str | None,
    bertscore_num_layers: int | None,
) -> dict[str, Any]:
    answer_map = {item["id"]: item for item in answers}
    results: list[dict[str, Any]] = []

    for question in questions:
        answer_record = answer_map[question.id]
        reference = question.reference_answer
        prediction = answer_record.get("generated_answer", "") or ""

        lexical = compute_lexical_metrics(reference=reference, prediction=prediction)
        result = {
            "id": question.id,
            "question": question.question,
            "reference_answer": reference,
            "generated_answer": prediction,
            "category": question.category,
            "difficulty": question.difficulty,
            "status": answer_record.get("status", "error"),
            "error": answer_record.get("error"),
            "model": answer_record.get("model"),
            "retrieval_count": answer_record.get("retrieval_count", 0),
            "latency_seconds": answer_record.get("latency_seconds", 0.0),
            "response_chars": len(normalize_text(prediction)),
            "reference_chars": len(normalize_text(reference)),
            "citations": answer_record.get("citations", []),
            **lexical,
        }
        results.append(result)

    bertscore_values = compute_bertscore_values(
        results=results,
        bertscore_model_ref=bertscore_model_ref,
        bertscore_num_layers=bertscore_num_layers,
    )
    for result, bertscore_f1 in zip(results, bertscore_values, strict=True):
        result["bertscore_f1"] = bertscore_f1

    return {
        "meta": {
            "computed_at": utc_now(),
            "question_count": len(results),
            "metric_keys": METRIC_KEYS,
            "bertscore_model_ref": bertscore_model_ref,
            "bertscore_num_layers": bertscore_num_layers,
        },
        "results": results,
    }


def compute_lexical_metrics(*, reference: str, prediction: str) -> dict[str, float]:
    ref_plain = normalize_text(reference)
    pred_plain = normalize_text(prediction)
    ref_tokens = tokenize_for_metrics(reference)
    pred_tokens = tokenize_for_metrics(prediction)

    bleu_1 = safe_bleu([ref_tokens], pred_tokens, weights=(1.0, 0.0, 0.0, 0.0))
    bleu_2 = safe_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0.0, 0.0))
    bleu_4 = safe_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    rouge_scores = safe_rouge(ref_tokens, pred_tokens)
    meteor = safe_meteor(ref_tokens, pred_tokens)
    chrf = safe_chrf(ref_plain, pred_plain)
    exact_match = 1.0 if normalize_for_exact_match(reference) == normalize_for_exact_match(prediction) else 0.0
    f1 = safe_f1(ref_tokens, pred_tokens)

    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4,
        "rouge_1_f": rouge_scores["rouge_1_f"],
        "rouge_2_f": rouge_scores["rouge_2_f"],
        "rouge_l_f": rouge_scores["rouge_l_f"],
        "meteor": meteor,
        "chrf": chrf,
        "exact_match": exact_match,
        "f1": f1,
    }


def safe_bleu(
    references: list[list[str]],
    prediction: list[str],
    *,
    weights: tuple[float, float, float, float],
) -> float:
    if not prediction or not any(reference for reference in references):
        return 0.0
    return round(sentence_bleu(references, prediction, weights=weights, smoothing_function=SMOOTHING), 4)


def safe_rouge(reference_tokens: list[str], prediction_tokens: list[str]) -> dict[str, float]:
    if not reference_tokens or not prediction_tokens:
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
    reference = " ".join(reference_tokens)
    prediction = " ".join(prediction_tokens)
    scores = ROUGE_SCORER.score(reference, prediction)
    return {
        "rouge_1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge_2_f": round(scores["rouge2"].fmeasure, 4),
        "rouge_l_f": round(scores["rougeL"].fmeasure, 4),
    }


def safe_meteor(reference_tokens: list[str], prediction_tokens: list[str]) -> float:
    if not reference_tokens or not prediction_tokens:
        return 0.0
    return round(
        meteor_score([reference_tokens], prediction_tokens, wordnet=EmptyWordNet()),
        4,
    )


def safe_chrf(reference: str, prediction: str) -> float:
    if not reference or not prediction:
        return 0.0
    return round(CHRF_SCORER.sentence_score(prediction, [reference]).score / 100.0, 4)


def safe_f1(reference_tokens: list[str], prediction_tokens: list[str]) -> float:
    if not reference_tokens or not prediction_tokens:
        return 0.0
    ref_counter = Counter(reference_tokens)
    pred_counter = Counter(prediction_tokens)
    overlap = sum((ref_counter & pred_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


def compute_bertscore_values(
    *,
    results: list[dict[str, Any]],
    bertscore_model_ref: str | None,
    bertscore_num_layers: int | None,
) -> list[float]:
    if bertscore_model_ref is None:
        return [0.0 for _ in results]

    predictions = [normalize_text(item["generated_answer"]) for item in results]
    references = [normalize_text(item["reference_answer"]) for item in results]

    with bertscore_env():
        try:
            from bert_score import score
        except ImportError:
            return [0.0 for _ in results]
        _, _, f1_scores = score(
            predictions,
            references,
            model_type=bertscore_model_ref,
            num_layers=bertscore_num_layers,
            verbose=False,
            batch_size=8,
        )
    return [round(float(value), 4) for value in f1_scores]


def build_summary_payload(
    *,
    metrics_payload: dict[str, Any],
    dataset_path: Path,
    answers_path: Path,
    base_url: str,
    provider: str,
    top_k: int,
    bertscore_model_ref: str | None,
    bertscore_num_layers: int | None,
) -> dict[str, Any]:
    results = metrics_payload["results"]
    aggregate = build_aggregate_summary(results)
    return {
        "meta": {
            "created_at": utc_now(),
            "dataset_path": str(dataset_path),
            "answers_path": str(answers_path),
            "base_url": base_url,
            "provider": provider,
            "top_k": top_k,
            "bertscore_model_ref": bertscore_model_ref,
            "bertscore_num_layers": bertscore_num_layers,
        },
        "aggregate": aggregate,
        "by_difficulty": build_group_summary(results, key="difficulty"),
        "by_category": build_group_summary(results, key="category"),
    }


def build_aggregate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summarize_metrics(results)
    ranking = sorted(
        (
            {"metric": metric, "mean": stats["mean"]}
            for metric, stats in metrics.items()
        ),
        key=lambda item: item["mean"],
        reverse=True,
    )
    success_count = sum(1 for item in results if item["status"] == "success")
    retrieval_counts = [int(item.get("retrieval_count") or 0) for item in results]
    latency_values = [float(item.get("latency_seconds") or 0.0) for item in results]
    return {
        "question_count": len(results),
        "success_count": success_count,
        "failed_count": len(results) - success_count,
        "success_rate": round(success_count / len(results), 4) if results else 0.0,
        "avg_retrieval_count": round(sum(retrieval_counts) / len(retrieval_counts), 4) if retrieval_counts else 0.0,
        "avg_latency_seconds": round(sum(latency_values) / len(latency_values), 4) if latency_values else 0.0,
        "metrics": metrics,
        "ranking_by_mean": ranking,
        "best_metric": ranking[0] if ranking else None,
    }


def build_group_summary(results: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(str(item[key]), []).append(item)

    summary: dict[str, Any] = {}
    for group_name, group_items in sorted(groups.items()):
        summary[group_name] = {
            "count": len(group_items),
            "success_count": sum(1 for item in group_items if item["status"] == "success"),
            "metrics": summarize_metrics(group_items),
        }
    return summary


def summarize_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRIC_KEYS:
        values = [float(item.get(metric, 0.0) or 0.0) for item in results]
        summary[metric] = {
            "mean": round(sum(values) / len(values), 4) if values else 0.0,
            "median": round(statistics.median(values), 4) if values else 0.0,
            "min": round(min(values), 4) if values else 0.0,
            "max": round(max(values), 4) if values else 0.0,
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        }
    return summary


def render_summary_markdown(summary_payload: dict[str, Any]) -> str:
    aggregate = summary_payload["aggregate"]
    lines = [
        "# Manual QA Evaluation Summary",
        "",
        f"- 生成时间：{summary_payload['meta']['created_at']}",
        f"- 问题数量：{aggregate['question_count']}",
        f"- 成功率：{aggregate['success_rate']:.2%}",
        f"- 平均检索条数：{aggregate['avg_retrieval_count']:.2f}",
        f"- 平均耗时：{aggregate['avg_latency_seconds']:.2f} 秒",
        f"- BERTScore 模型：{summary_payload['meta']['bertscore_model_ref'] or '未启用'}",
        "",
        "## Overall Ranking",
        "",
        "| Metric | Mean | Median | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregate["ranking_by_mean"]:
        metric = item["metric"]
        stats = aggregate["metrics"][metric]
        lines.append(
            f"| {metric} | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## By Difficulty",
            "",
            "| Difficulty | Count | BLEU-1 | ROUGE-L | METEOR | BERTScore | ChrF | EM | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for difficulty, payload in summary_payload["by_difficulty"].items():
        metrics = payload["metrics"]
        lines.append(
            f"| {difficulty} | {payload['count']} | {metrics['bleu_1']['mean']:.4f} | "
            f"{metrics['rouge_l_f']['mean']:.4f} | {metrics['meteor']['mean']:.4f} | "
            f"{metrics['bertscore_f1']['mean']:.4f} | {metrics['chrf']['mean']:.4f} | "
            f"{metrics['exact_match']['mean']:.4f} | {metrics['f1']['mean']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_metrics_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "id",
        "category",
        "difficulty",
        "status",
        "model",
        "retrieval_count",
        "latency_seconds",
        "response_chars",
        "reference_chars",
        *METRIC_KEYS,
        "question",
        "generated_answer",
        "reference_answer",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name) for name in fieldnames})


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("`", "")
    normalized = normalized.replace("**", "")
    normalized = normalized.replace("__", "")
    normalized = re.sub(r"^\s*#{1,6}\s*", "", normalized, flags=re.MULTILINE)
    normalized = MARKDOWN_BULLET_PATTERN.sub("", normalized)
    normalized = normalized.replace("|", " ")
    normalized = normalized.replace(">", " ")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip().lower()


def normalize_for_exact_match(text: str) -> str:
    return "".join(tokenize_for_metrics(text))


def tokenize_for_metrics(text: str) -> list[str]:
    normalized = normalize_text(text)
    return TOKEN_PATTERN.findall(normalized)


def resolve_bertscore_model(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if BGE_SMALL_ZH_CACHE_DIR.exists():
        snapshots = sorted(path for path in BGE_SMALL_ZH_CACHE_DIR.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return None


def resolve_bertscore_num_layers(model_ref: str | None, *, explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    if model_ref is None:
        return None
    model_path = Path(model_ref)
    if model_path.exists():
        config_path = model_path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            num_hidden_layers = config.get("num_hidden_layers")
            if isinstance(num_hidden_layers, int):
                return num_hidden_layers
    return None


@contextlib.contextmanager
def bertscore_env() -> Iterator[None]:
    temp_root = Path("/private/tmp/codex_bertscore")
    temp_root.mkdir(parents=True, exist_ok=True)
    env_updates = {
        "MPLCONFIGDIR": str(temp_root / "mpl"),
        "XDG_CACHE_HOME": str(temp_root / "xdg"),
        "NO_PROXY": "*",
        "no_proxy": "*",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
    }
    snapshot = {key: os.environ.get(key) for key in env_updates}
    for key, value in env_updates.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key, original in snapshot.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise
