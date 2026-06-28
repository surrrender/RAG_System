"""
标注结果指标计算工具
======================

用法:
  python -m evaluation.compute_annotation_metrics outputs/annotation_results_no_reranker.json

功能:
  从人工标注结果 JSON 计算以下指标：
    - recall@1, @3, @5, @7, @10
    - MRR (Mean Reciprocal Rank)
    - Hit Rate@1, @3, @5, @7, @10
    - 按难度、分类的分组指标
    - 标注统计信息
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_recall_at_k(
    relevant_ids: set[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(relevant_ids & top_k)
    return hits / len(relevant_ids)


def compute_mrr(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_hit_rate(
    relevant_ids: set[str],
    retrieved_ids: list[str],
    k: int,
) -> bool:
    if not relevant_ids:
        return True
    top_k = set(retrieved_ids[:k])
    return bool(relevant_ids & top_k)


def load_annotation_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_results(data: dict) -> dict:
    config = data.get("config", {})
    annotations = data.get("annotations", [])

    K_VALUES = [1, 3, 5, 7, 10]
    per_question = []
    all_recalls = {k: [] for k in K_VALUES}
    all_mrrs = []
    zero_relevant_count = 0

    for ann in annotations:
        qid = ann["question_id"]
        question = ann.get("question", "")
        category = ann.get("category", "")
        difficulty = ann.get("difficulty", "")

        relevant = set(ann.get("relevant_chunk_ids", []))
        retrieved = [c["chunk_id"] for c in ann.get("all_retrieved_chunks", [])]

        if len(relevant) == 0:
            zero_relevant_count += 1

        rec = {}
        for k in K_VALUES:
            r = compute_recall_at_k(relevant, retrieved, k)
            rec[f"recall@{k}"] = r
            all_recalls[k].append(r)

        mrr_val = compute_mrr(relevant, retrieved)
        all_mrrs.append(mrr_val)

        hit = {}
        for k in K_VALUES:
            hit[f"hit@{k}"] = compute_hit_rate(relevant, retrieved, k)

        per_question.append({
            "question_id": qid,
            "question": question[:80] + ("..." if len(question) > 80 else ""),
            "category": category,
            "difficulty": difficulty,
            "relevant_count": len(relevant),
            **rec,
            "mrr": mrr_val,
            **hit,
        })

    n = max(len(per_question), 1)
    summary = {
        "config": config,
        "total_questions": len(annotations),
        "zero_relevant_count": zero_relevant_count,
        "avg_recall": {f"@{k}": round(sum(v) / n, 4) for k, v in all_recalls.items()},
        "avg_mrr": round(sum(all_mrrs) / n, 4),
        "hit_rate": {
            f"@{k}": round(sum(1 for q in per_question if q[f"hit@{k}"]) / n, 4)
            for k in K_VALUES
        },
    }

    # Group by difficulty
    by_difficulty: dict[str, dict] = {}
    for q in per_question:
        d = q["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"count": 0, "recalls": {k: [] for k in K_VALUES}, "mrrs": []}
        by_difficulty[d]["count"] += 1
        for k in K_VALUES:
            by_difficulty[d]["recalls"][k].append(q[f"recall@{k}"])
        by_difficulty[d]["mrrs"].append(q["mrr"])

    summary["by_difficulty"] = {}
    for d, v in by_difficulty.items():
        c = max(v["count"], 1)
        summary["by_difficulty"][d] = {
            "count": v["count"],
            "avg_recall": {f"@{k}": round(sum(v["recalls"][k]) / c, 4) for k in K_VALUES},
            "avg_mrr": round(sum(v["mrrs"]) / c, 4),
        }

    # Group by category
    by_category: dict[str, dict] = {}
    for q in per_question:
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "recalls": {k: [] for k in K_VALUES}, "mrrs": []}
        by_category[cat]["count"] += 1
        for k in K_VALUES:
            by_category[cat]["recalls"][k].append(q[f"recall@{k}"])
        by_category[cat]["mrrs"].append(q["mrr"])

    summary["by_category"] = {}
    for cat, v in by_category.items():
        c = max(v["count"], 1)
        summary["by_category"][cat] = {
            "count": v["count"],
            "avg_recall": {f"@{k}": round(sum(v["recalls"][k]) / c, 4) for k in K_VALUES},
            "avg_mrr": round(sum(v["mrrs"]) / c, 4),
        }

    summary["per_question"] = per_question
    return summary


def print_report(summary: dict) -> None:
    config = summary["config"]
    suffix = "（启用 Reranker）" if config.get("reranker_enabled") else "（未启用 Reranker）"

    print("=" * 65)
    print(f"  检索召回评估报告 {suffix}")
    print("=" * 65)
    print(f"  问题总数: {summary['total_questions']}")
    print(f"  标注为空（找不到相关 chunk）: {summary['zero_relevant_count']} 题")
    print(f"  生成时间: {config.get('generated_at', 'N/A')}")
    print()

    print("-" * 65)
    print(f"  {'指标':<20} {'数值':<10}")
    print("-" * 65)
    for k, v in summary["avg_recall"].items():
        print(f"  {'Recall' + k:<20} {v:<10.2%}")
    print(f"  {'MRR':<20} {summary['avg_mrr']:<10.4f}")
    print("-" * 65)
    for k, v in summary["hit_rate"].items():
        print(f"  {'Hit Rate' + k:<20} {v:<10.2%}")
    print("-" * 65)
    print()

    if summary["by_difficulty"]:
        print("按难度分组:")
        print(f"  {'难度':<16} {'数量':<6} {'R@5':<10} {'R@10':<10} {'MRR':<10}")
        print("  " + "-" * 52)
        for d in ["easy", "medium", "hard"]:
            if d in summary["by_difficulty"]:
                v = summary["by_difficulty"][d]
                print(f"  {d:<16} {v['count']:<6} {v['avg_recall']['@5']:<10.2%} {v['avg_recall']['@10']:<10.2%} {v['avg_mrr']:<10.4f}")
        print()

    if summary["by_category"]:
        print("按分类分组:")
        print(f"  {'分类':<20} {'数量':<6} {'R@5':<10} {'R@10':<10} {'MRR':<10}")
        print("  " + "-" * 56)
        for cat, v in sorted(summary["by_category"].items(), key=lambda x: -x[1]["count"]):
            cat_short = cat[:18]
            print(f"  {cat_short:<20} {v['count']:<6} {v['avg_recall']['@5']:<10.2%} {v['avg_recall']['@10']:<10.2%} {v['avg_mrr']:<10.4f}")
        print()


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m evaluation.compute_annotation_metrics <annotation_results.json>")
        print("示例: python -m evaluation.compute_annotation_metrics outputs/annotation_results_no_reranker.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        sys.exit(1)

    print(f"加载标注结果: {input_path}")
    data = load_annotation_results(input_path)
    summary = analyze_results(data)

    # Save summary
    output_path = OUTPUT_DIR / f"metrics_{input_path.stem}.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_report(summary)
    print(f"详细结果已保存: {output_path.resolve()}")


if __name__ == "__main__":
    main()
