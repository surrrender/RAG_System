from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from embedding_indexing.io import iter_chunks
from evaluation.evaluators.metrics import hit_rate, mrr, recall_at_k
from evaluation.generators.question_generator import TestQuestion
from evaluation.strategies.bm25 import build_bm25_index
from evaluation.strategies.hybrid import HybridRetriever, StrategyResult
from llm.config import load_settings
from llm.retrieval import Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
CHUNKS_JSONL = REPO_ROOT / "Crawler" / "outputs" / "framework_chunks.jsonl"
QUESTIONS_FILE = EVAL_DIR / "data" / "manual_questions.json"
ANNOTATION_FILE = REPO_ROOT / "outputs" / "annotation_results_with_reranker_optimize.json"
OUTPUT_DIR = EVAL_DIR / "outputs"
OUTPUT_JSON = OUTPUT_DIR / "strategy_comparison_55_manual.json"
OUTPUT_HTML = OUTPUT_DIR / "strategy_comparison_55_manual.html"
K_VALUES = [1, 3, 5, 10]
STRATEGY_ORDER = ["Dense", "BM25", "Hybrid"]


@dataclass(slots=True)
class StrategyQuestionResult:
    question_id: str
    question: str
    category: str
    difficulty: str
    relevant_chunk_ids: list[str]
    strategy: str
    retrieved_chunk_ids: list[str]
    scores: list[float]
    total_ms: float
    recall: dict[int, float]
    hit_rate: dict[int, bool]
    mrr: float


@dataclass(slots=True)
class StrategySummary:
    strategy: str
    question_count: int = 0
    avg_recall: dict[int, float] = field(default_factory=dict)
    avg_hit_rate: dict[int, float] = field(default_factory=dict)
    avg_mrr: float = 0.0
    avg_total_ms: float = 0.0
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)


def load_manual_questions_with_annotations() -> tuple[list[TestQuestion], dict[str, object]]:
    questions_data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    annotation_data = json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))

    annotations_by_id = {item["question_id"]: item for item in annotation_data["annotations"]}
    merged_questions: list[TestQuestion] = []
    missing_annotations: list[str] = []
    zero_relevant_ids: list[str] = []

    for q in questions_data["questions"]:
        qid = q["id"]
        annotation = annotations_by_id.get(qid)
        if annotation is None:
            missing_annotations.append(qid)
            continue

        relevant_chunk_ids = list(annotation.get("relevant_chunk_ids", []))
        if not relevant_chunk_ids:
            zero_relevant_ids.append(qid)
            continue

        merged_questions.append(
            TestQuestion(
                id=qid,
                question=q["question"],
                relevant_chunk_ids=relevant_chunk_ids,
                reference_answer=q.get("reference_answer", ""),
                category=q.get("category") or annotation.get("category", "general"),
                difficulty=q.get("difficulty") or annotation.get("difficulty", "medium"),
                source_doc_title=q.get("source_doc_title", ""),
            )
        )

    meta = {
        "manual_total_questions": len(questions_data["questions"]),
        "annotation_total_questions": len(annotation_data["annotations"]),
        "evaluable_questions": len(merged_questions),
        "skipped_zero_relevant_questions": zero_relevant_ids,
        "missing_annotation_questions": missing_annotations,
        "annotation_config": annotation_data.get("config", {}),
        "annotation_created_at": annotation_data.get("config", {}).get("generated_at"),
    }
    return merged_questions, meta


def evaluate_questions(hybrid_retriever: HybridRetriever, questions: list[TestQuestion], top_k: int = 10) -> list[StrategyQuestionResult]:
    results: list[StrategyQuestionResult] = []

    for question in tqdm(questions, desc="5.5 strategy comparison"):
        relevant_ids = set(question.relevant_chunk_ids)
        strategy_results: list[StrategyResult] = [
            hybrid_retriever.search_dense_only(question.question, top_k=top_k),
            hybrid_retriever.search_bm25_only(question.question, top_k=top_k),
            hybrid_retriever.search_hybrid(question.question, top_k=top_k),
        ]

        for strategy_result in strategy_results:
            retrieved_ids = strategy_result.chunk_ids
            recall = {k: recall_at_k(relevant_ids, retrieved_ids, k) for k in K_VALUES}
            hits = {k: hit_rate(relevant_ids, retrieved_ids, k) for k in K_VALUES}
            results.append(
                StrategyQuestionResult(
                    question_id=question.id,
                    question=question.question,
                    category=question.category,
                    difficulty=question.difficulty,
                    relevant_chunk_ids=question.relevant_chunk_ids,
                    strategy=strategy_result.strategy_name,
                    retrieved_chunk_ids=retrieved_ids,
                    scores=strategy_result.scores,
                    total_ms=strategy_result.timing.get("total_ms", 0.0),
                    recall=recall,
                    hit_rate=hits,
                    mrr=mrr(relevant_ids, retrieved_ids),
                )
            )

    return results


def summarize_results(results: list[StrategyQuestionResult]) -> dict[str, StrategySummary]:
    grouped: dict[str, list[StrategyQuestionResult]] = defaultdict(list)
    for result in results:
        grouped[result.strategy].append(result)

    summaries: dict[str, StrategySummary] = {}
    for strategy in STRATEGY_ORDER:
        items = grouped.get(strategy, [])
        count = len(items)
        if count == 0:
            summaries[strategy] = StrategySummary(strategy=strategy)
            continue

        summary = StrategySummary(
            strategy=strategy,
            question_count=count,
            avg_recall={k: round(sum(item.recall[k] for item in items) / count, 4) for k in K_VALUES},
            avg_hit_rate={k: round(sum(1 for item in items if item.hit_rate[k]) / count, 4) for k in K_VALUES},
            avg_mrr=round(sum(item.mrr for item in items) / count, 4),
            avg_total_ms=round(sum(item.total_ms for item in items) / count, 3),
        )

        by_difficulty: dict[str, list[StrategyQuestionResult]] = defaultdict(list)
        for item in items:
            by_difficulty[item.difficulty].append(item)

        summary.by_difficulty = {
            difficulty: {
                "count": len(diff_items),
                "recall@5": round(sum(item.recall[5] for item in diff_items) / len(diff_items), 4),
                "recall@10": round(sum(item.recall[10] for item in diff_items) / len(diff_items), 4),
                "mrr": round(sum(item.mrr for item in diff_items) / len(diff_items), 4),
            }
            for difficulty, diff_items in by_difficulty.items()
        }
        summaries[strategy] = summary

    return summaries


def build_dataset_stats(questions: list[TestQuestion], meta: dict[str, object]) -> dict[str, object]:
    all_chunks = list(iter_chunks(CHUNKS_JSONL))
    return {
        "total_documents": len(set(chunk.doc_id for chunk in all_chunks)),
        "total_chunks": len(all_chunks),
        "manual_total_questions": meta["manual_total_questions"],
        "annotation_total_questions": meta["annotation_total_questions"],
        "evaluable_questions": meta["evaluable_questions"],
        "skipped_zero_relevant_count": len(meta["skipped_zero_relevant_questions"]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": [
            {
                "id": question.id,
                "question": question.question,
                "category": question.category,
                "difficulty": question.difficulty,
                "relevant_count": len(question.relevant_chunk_ids),
            }
            for question in questions
        ],
    }


def build_dense_retriever(settings, qdrant_path: Path | None = None) -> Retriever:
    return Retriever(
        qdrant_path=qdrant_path or settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        embedder_provider=settings.embedder_provider,
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        reranker_provider=settings.reranker_provider,
        reranker_model=settings.reranker_model,
        reranker_device=settings.reranker_device,
        rerank_candidate_limit=15,
        disable_reranker=True,
    )


def warm_up_dense_retriever(settings) -> tuple[Retriever, Path | None]:
    retriever = build_dense_retriever(settings)
    try:
        retriever.warm_up()
        return retriever, None
    except RuntimeError as exc:
        message = str(exc)
        if "already accessed by another instance of Qdrant client" not in message:
            raise

    temp_qdrant_dir = Path(tempfile.mkdtemp(prefix="rag_eval_qdrant_55.", dir="/private/tmp"))
    print(f"Detected locked local Qdrant. Copying index to {temp_qdrant_dir} ...")
    shutil.copytree(settings.qdrant_path, temp_qdrant_dir, dirs_exist_ok=True)

    retry_retriever = build_dense_retriever(settings, qdrant_path=temp_qdrant_dir)
    retry_retriever.warm_up()
    return retry_retriever, temp_qdrant_dir


def render_html_report(
    output_path: Path,
    dataset_stats: dict[str, object],
    summaries: dict[str, StrategySummary],
    question_results: list[StrategyQuestionResult],
) -> None:
    questions_by_id: dict[str, dict[str, StrategyQuestionResult]] = defaultdict(dict)
    question_meta: dict[str, StrategyQuestionResult] = {}
    for result in question_results:
        questions_by_id[result.question_id][result.strategy] = result
        question_meta[result.question_id] = result

    best_recall_strategy = max(STRATEGY_ORDER, key=lambda name: summaries[name].avg_recall.get(5, 0.0))
    best_mrr_strategy = max(STRATEGY_ORDER, key=lambda name: summaries[name].avg_mrr)
    dense_recall_5 = summaries["Dense"].avg_recall.get(5, 0.0)
    hybrid_recall_5 = summaries["Hybrid"].avg_recall.get(5, 0.0)
    hybrid_vs_dense = ((hybrid_recall_5 - dense_recall_5) / dense_recall_5 * 100) if dense_recall_5 else 0.0

    summary_rows = "".join(
        f"""
        <tr>
            <td><strong>{summary.strategy}</strong></td>
            <td>{summary.avg_recall.get(1, 0.0):.1%}</td>
            <td>{summary.avg_recall.get(3, 0.0):.1%}</td>
            <td>{summary.avg_recall.get(5, 0.0):.1%}</td>
            <td>{summary.avg_recall.get(10, 0.0):.1%}</td>
            <td>{summary.avg_mrr:.4f}</td>
            <td>{summary.avg_hit_rate.get(5, 0.0):.1%}</td>
            <td>{summary.avg_total_ms:.1f}ms</td>
        </tr>
        """
        for summary in (summaries[name] for name in STRATEGY_ORDER)
    )

    difficulty_order = ["easy", "medium", "hard"]
    difficulty_rows = ""
    for difficulty in difficulty_order:
        if not any(difficulty in summaries[name].by_difficulty for name in STRATEGY_ORDER):
            continue
        row_cells = []
        counts = []
        for name in STRATEGY_ORDER:
            item = summaries[name].by_difficulty.get(difficulty)
            if item is None:
                counts.append("-")
                row_cells.extend(["-", "-", "-"])
            else:
                counts.append(str(item["count"]))
                row_cells.extend([
                    f"{item['recall@5']:.1%}",
                    f"{item['recall@10']:.1%}",
                    f"{item['mrr']:.4f}",
                ])
        difficulty_rows += (
            "<tr>"
            f"<td>{difficulty}</td>"
            f"<td>{counts[0]}</td><td>{row_cells[0]}</td><td>{row_cells[1]}</td><td>{row_cells[2]}</td>"
            f"<td>{counts[1]}</td><td>{row_cells[3]}</td><td>{row_cells[4]}</td><td>{row_cells[5]}</td>"
            f"<td>{counts[2]}</td><td>{row_cells[6]}</td><td>{row_cells[7]}</td><td>{row_cells[8]}</td>"
            "</tr>"
        )

    per_question_rows = ""
    for question_id in sorted(questions_by_id):
        meta = question_meta[question_id]
        dense = questions_by_id[question_id]["Dense"]
        bm25 = questions_by_id[question_id]["BM25"]
        hybrid = questions_by_id[question_id]["Hybrid"]
        per_question_rows += f"""
        <tr>
            <td>{question_id}</td>
            <td class="question-cell" title="{meta.question}">{meta.question}</td>
            <td>{meta.difficulty}</td>
            <td>{len(meta.relevant_chunk_ids)}</td>
            <td>{dense.recall[5]:.1%}</td>
            <td>{dense.mrr:.3f}</td>
            <td>{bm25.recall[5]:.1%}</td>
            <td>{bm25.mrr:.3f}</td>
            <td>{hybrid.recall[5]:.1%}</td>
            <td>{hybrid.mrr:.3f}</td>
        </tr>
        """

    chart_labels = json.dumps([f"K={k}" for k in K_VALUES], ensure_ascii=False)
    dense_recall = json.dumps([summaries["Dense"].avg_recall.get(k, 0.0) for k in K_VALUES])
    bm25_recall = json.dumps([summaries["BM25"].avg_recall.get(k, 0.0) for k in K_VALUES])
    hybrid_recall = json.dumps([summaries["Hybrid"].avg_recall.get(k, 0.0) for k in K_VALUES])
    mrr_labels = json.dumps(STRATEGY_ORDER, ensure_ascii=False)
    mrr_values = json.dumps([summaries[name].avg_mrr for name in STRATEGY_ORDER])
    latency_values = json.dumps([summaries[name].avg_total_ms for name in STRATEGY_ORDER])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>5.5 不同检索策略对比实验</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif; background: #f3f6fb; color: #10233f; }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%); color: #fff; border-radius: 20px; padding: 32px; margin-bottom: 24px; box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18); }}
    .hero h1 {{ margin: 0 0 10px; font-size: 32px; }}
    .hero p {{ margin: 6px 0; color: rgba(255, 255, 255, 0.88); }}
    .section {{ background: #fff; border-radius: 18px; padding: 24px; margin-bottom: 24px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); }}
    .section h2 {{ margin: 0 0 18px; font-size: 22px; color: #0f172a; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
    .stat-card {{ border-radius: 16px; padding: 18px; background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%); border: 1px solid #dbe7ff; }}
    .stat-card .label {{ color: #4a5f7d; font-size: 13px; margin-bottom: 8px; }}
    .stat-card .value {{ font-size: 30px; font-weight: 700; color: #0f172a; }}
    .analysis-note {{ margin-top: 16px; padding: 14px 16px; border-radius: 14px; background: #f8fafc; color: #334155; line-height: 1.7; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid #e8eef7; text-align: left; font-size: 14px; }}
    th {{ background: #f8fafc; color: #334155; font-weight: 600; }}
    tr:hover td {{ background: #fbfdff; }}
    .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .chart-card {{ background: #f8fafc; border: 1px solid #e5edf8; border-radius: 16px; padding: 16px; }}
    .chart-card h3 {{ margin: 0 0 12px; font-size: 16px; color: #1e293b; }}
    .question-cell {{ min-width: 320px; max-width: 420px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .footnote {{ color: #64748b; font-size: 13px; margin-top: 14px; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>5.5 不同检索策略对比实验</h1>
      <p>对比策略：Dense / BM25 / Hybrid</p>
      <p>问题集：{dataset_stats['manual_total_questions']} 题人工测试集，实际纳入评测 {dataset_stats['evaluable_questions']} 题</p>
      <p>相关 chunk 标注来源：annotation_results_with_reranker_optimize.json</p>
    </div>

    <div class="section">
      <h2>实验设置</h2>
      <div class="stats-grid">
        <div class="stat-card"><div class="label">文档数量</div><div class="value">{dataset_stats['total_documents']}</div></div>
        <div class="stat-card"><div class="label">Chunk 数量</div><div class="value">{dataset_stats['total_chunks']}</div></div>
        <div class="stat-card"><div class="label">人工问题总数</div><div class="value">{dataset_stats['manual_total_questions']}</div></div>
        <div class="stat-card"><div class="label">纳入评测问题数</div><div class="value">{dataset_stats['evaluable_questions']}</div></div>
        <div class="stat-card"><div class="label">跳过问题数</div><div class="value">{dataset_stats['skipped_zero_relevant_count']}</div></div>
      </div>
      <div class="analysis-note">
        本次实验按你的要求使用 <code>evaluation/data/manual_questions.json</code> 作为问题集，
        并从 <code>outputs/annotation_results_with_reranker_optimize.json</code> 读取每题的人工相关 chunk 标注。
        对于“人工标注没有找到任何相关 chunk”的题目，本报告将其从平均指标中剔除，避免把不可评估样本计成策略优势。
      </div>
    </div>

    <div class="section">
      <h2>核心结果</h2>
      <div class="stats-grid">
        <div class="stat-card"><div class="label">最佳 Recall@5</div><div class="value">{summaries[best_recall_strategy].avg_recall.get(5, 0.0):.1%}</div></div>
        <div class="stat-card"><div class="label">最佳 Recall@5 策略</div><div class="value">{best_recall_strategy}</div></div>
        <div class="stat-card"><div class="label">最佳 MRR 策略</div><div class="value">{best_mrr_strategy}</div></div>
        <div class="stat-card"><div class="label">Hybrid 相对 Dense 提升</div><div class="value">{hybrid_vs_dense:+.1f}%</div></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>策略</th>
              <th>Recall@1</th>
              <th>Recall@3</th>
              <th>Recall@5</th>
              <th>Recall@10</th>
              <th>MRR</th>
              <th>Hit@5</th>
              <th>平均耗时</th>
            </tr>
          </thead>
          <tbody>{summary_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <h2>图表对比</h2>
      <div class="charts-grid">
        <div class="chart-card">
          <h3>Recall@K 曲线</h3>
          <canvas id="recallChart"></canvas>
        </div>
        <div class="chart-card">
          <h3>MRR 对比</h3>
          <canvas id="mrrChart"></canvas>
        </div>
        <div class="chart-card">
          <h3>平均检索耗时</h3>
          <canvas id="latencyChart"></canvas>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>按难度分组</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>难度</th>
              <th>Dense 数量</th><th>Dense R@5</th><th>Dense R@10</th><th>Dense MRR</th>
              <th>BM25 数量</th><th>BM25 R@5</th><th>BM25 R@10</th><th>BM25 MRR</th>
              <th>Hybrid 数量</th><th>Hybrid R@5</th><th>Hybrid R@10</th><th>Hybrid MRR</th>
            </tr>
          </thead>
          <tbody>{difficulty_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <h2>逐题结果</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>问题</th>
              <th>难度</th>
              <th>相关 Chunk 数</th>
              <th>Dense R@5</th>
              <th>Dense MRR</th>
              <th>BM25 R@5</th>
              <th>BM25 MRR</th>
              <th>Hybrid R@5</th>
              <th>Hybrid MRR</th>
            </tr>
          </thead>
          <tbody>{per_question_rows}</tbody>
        </table>
      </div>
      <div class="footnote">
        报告生成时间：{dataset_stats['created_at']}。如果你后续想把图表直接放论文里，建议优先使用本页的核心结果表和 Recall@K 曲线图。
      </div>
    </div>
  </div>

  <script>
    new Chart(document.getElementById('recallChart'), {{
      type: 'line',
      data: {{
        labels: {chart_labels},
        datasets: [
          {{ label: 'Dense', data: {dense_recall}, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)', tension: 0.25, fill: false }},
          {{ label: 'BM25', data: {bm25_recall}, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.08)', tension: 0.25, fill: false }},
          {{ label: 'Hybrid', data: {hybrid_recall}, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)', tension: 0.25, fill: false }}
        ]
      }},
      options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
    }});

    new Chart(document.getElementById('mrrChart'), {{
      type: 'bar',
      data: {{
        labels: {mrr_labels},
        datasets: [
          {{ label: 'MRR', data: {mrr_values}, backgroundColor: ['#2563eb', '#f59e0b', '#10b981'] }}
        ]
      }},
      options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
    }});

    new Chart(document.getElementById('latencyChart'), {{
      type: 'bar',
      data: {{
        labels: {mrr_labels},
        datasets: [
          {{ label: 'Average latency (ms)', data: {latency_values}, backgroundColor: ['#93c5fd', '#fcd34d', '#86efac'] }}
        ]
      }},
      options: {{ responsive: true }}
    }});
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    if not QUESTIONS_FILE.exists():
        print(f"Questions file not found: {QUESTIONS_FILE}")
        sys.exit(1)
    if not ANNOTATION_FILE.exists():
        print(f"Annotation file not found: {ANNOTATION_FILE}")
        sys.exit(1)
    if not CHUNKS_JSONL.exists():
        print(f"Chunks file not found: {CHUNKS_JSONL}")
        sys.exit(1)

    t0 = time.perf_counter()
    print("=" * 60)
    print("  Section 5.5 Strategy Comparison")
    print("  Strategies: Dense / BM25 / Hybrid")
    print("=" * 60)

    questions, meta = load_manual_questions_with_annotations()
    print(f"Manual questions: {meta['manual_total_questions']}")
    print(f"Annotated questions: {meta['annotation_total_questions']}")
    print(f"Evaluable questions: {meta['evaluable_questions']}")
    print(f"Skipped zero-relevant questions: {len(meta['skipped_zero_relevant_questions'])}")

    settings = load_settings()

    print("\nBuilding BM25 index...")
    bm25 = build_bm25_index(str(CHUNKS_JSONL))
    print(f"BM25 indexed {bm25._total_docs} documents")

    print("\nLoading dense retriever...")
    dense_retriever, temp_qdrant_dir = warm_up_dense_retriever(settings)
    if temp_qdrant_dir is not None:
        print(f"Using temporary Qdrant copy: {temp_qdrant_dir}")

    hybrid_retriever = HybridRetriever(
        embedder=dense_retriever._embedder,
        qdrant_index=dense_retriever._index,
        bm25=bm25,
        reranker=None,
        rerank_candidate_limit=15,
    )

    question_results = evaluate_questions(hybrid_retriever, questions, top_k=10)
    summaries = summarize_results(question_results)
    dataset_stats = build_dataset_stats(questions, meta)

    payload = {
        "dataset_stats": dataset_stats,
        "meta": meta,
        "effective_qdrant_path": str(temp_qdrant_dir or settings.qdrant_path),
        "summaries": {name: asdict(summary) for name, summary in summaries.items()},
        "per_question": [asdict(result) for result in question_results],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html_report(OUTPUT_HTML, dataset_stats, summaries, question_results)

    print("\nSummary:")
    for name in STRATEGY_ORDER:
        summary = summaries[name]
        print(
            f"  {name:<7} "
            f"R@5={summary.avg_recall.get(5, 0.0):.1%}  "
            f"R@10={summary.avg_recall.get(10, 0.0):.1%}  "
            f"MRR={summary.avg_mrr:.4f}  "
            f"Hit@5={summary.avg_hit_rate.get(5, 0.0):.1%}  "
            f"{summary.avg_total_ms:.1f}ms"
        )

    print(f"\nJSON: {OUTPUT_JSON.resolve()}")
    print(f"HTML: {OUTPUT_HTML.resolve()}")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")

    try:
        if dense_retriever._index is not None:
            dense_retriever._index.client.close()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
