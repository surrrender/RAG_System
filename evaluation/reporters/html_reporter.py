from __future__ import annotations

import json
import time
from pathlib import Path

from evaluation.evaluators.retrieval_evaluator import RetrievalEvalResult, PerQuestionRetrievalResult
from evaluation.evaluators.qa_evaluator import QAEvalResult, PerQuestionQAResult


def generate_html_report(
    output_path: Path,
    dataset_stats: dict,
    retrieval_result: RetrievalEvalResult | None,
    qa_result: QAEvalResult | None,
) -> None:
    sections: list[str] = []

    sections.append(_build_header(dataset_stats))
    sections.append(_build_dataset_section(dataset_stats))

    if retrieval_result:
        sections.append(_build_retrieval_section(retrieval_result))
    if qa_result and retrieval_result:
        sections.append(_build_qa_section(qa_result))
        sections.append(_build_analysis_section(retrieval_result, qa_result))

    body = "\n".join(sections)
    html = _WRAPPER.format(
        title="RAG 系统评估报告 — 微信小程序开发文档",
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        body=body,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _build_header(stats: dict) -> str:
    return f"""
    <div class="header">
        <h1>RAG 系统评估报告</h1>
        <p class="subtitle">微信小程序开发文档 · {stats.get('model', 'unknown')} · {stats.get('created_at', '')}</p>
    </div>
    """


def _build_dataset_section(stats: dict) -> str:
    questions = stats.get("questions", [])
    categories: dict[str, int] = {}
    difficulty: dict[str, int] = {}
    for q in questions:
        cat = q.get("category", "other")
        diff = q.get("difficulty", "medium")
        categories[cat] = categories.get(cat, 0) + 1
        difficulty[diff] = difficulty.get(diff, 0) + 1

    cat_rows = "".join(f"<tr><td>{c}</td><td>{n}</td></tr>" for c, n in sorted(categories.items(), key=lambda x: -x[1]))
    diff_rows = "".join(
        f"<tr><td>{d}</td><td>{n}</td></tr>" for d, n in sorted(difficulty.items(), key=lambda x: -x[1])
    )

    cat_labels = json.dumps(list(dict(sorted(categories.items(), key=lambda x: -x[1])).keys()), ensure_ascii=False)
    cat_data = json.dumps(list(dict(sorted(categories.items(), key=lambda x: -x[1])).values()))
    diff_labels = json.dumps(list(difficulty.keys()), ensure_ascii=False)
    diff_data = json.dumps(list(difficulty.values()))

    return f"""
    <div class="section">
        <h2>1. 数据集统计 (5.2)</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_documents', 0):,}</div>
                <div class="stat-label">技术文档数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_chunks', 0):,}</div>
                <div class="stat-label">文档 Chunk 数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_questions', 0)}</div>
                <div class="stat-label">测试 QA 对数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">1:1</div>
                <div class="stat-label">QA:Chunk 对应关系</div>
            </div>
        </div>
        <div class="charts-row">
            <div class="chart-box">
                <h3>问题类别分布</h3>
                <canvas id="categoryChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>问题难度分布</h3>
                <canvas id="difficultyChart"></canvas>
            </div>
        </div>
        <div class="tables-row">
            <div>
                <h3>类别统计</h3>
                <table><thead><tr><th>类别</th><th>数量</th></tr></thead><tbody>{cat_rows}</tbody></table>
            </div>
            <div>
                <h3>难度统计</h3>
                <table><thead><tr><th>难度</th><th>数量</th></tr></thead><tbody>{diff_rows}</tbody></table>
            </div>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('categoryChart'), {{
        type: 'pie',
        data: {{ labels: {cat_labels}, datasets: [{{ data: {cat_data}, backgroundColor: ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899','#06b6d4','#f97316'] }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});
    new Chart(document.getElementById('difficultyChart'), {{
        type: 'doughnut',
        data: {{ labels: {diff_labels}, datasets: [{{ data: {diff_data}, backgroundColor: ['#10b981','#f59e0b','#ef4444'] }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});
    </script>
    """


def _build_retrieval_section(r: RetrievalEvalResult) -> str:
    detail_rows = ""
    for q in r.questions[:50]:
        detail_rows += f"""
        <tr>
            <td>{q.question_id}</td>
            <td class="q-text" title="{_esc(q.question[:80])}">{_esc(q.question[:50])}...</td>
            <td>{q.category}</td>
            <td><span class="badge badge-{q.difficulty}">{q.difficulty}</span></td>
            <td>{q.recall_at_1:.2f}</td>
            <td>{q.recall_at_5:.2f}</td>
            <td>{q.recall_at_10:.2f}</td>
            <td>{q.mrr:.3f}</td>
            <td>{'Y' if q.hit_at_5 else 'N'}</td>
            <td>{q.retrieval_scores[0]:.4f}</td>
        </tr>
        """

    return f"""
    <div class="section">
        <h2>2. 检索效果评估 (5.3)</h2>

        <div class="stats-grid">
            <div class="stat-card accent-blue">
                <div class="stat-value">{r.avg_recall_at_5:.2%}</div>
                <div class="stat-label">Recall@5</div>
            </div>
            <div class="stat-card accent-green">
                <div class="stat-value">{r.avg_mrr:.3f}</div>
                <div class="stat-label">MRR</div>
            </div>
            <div class="stat-card accent-orange">
                <div class="stat-value">{r.hit_rate_at_5:.1%}</div>
                <div class="stat-label">Hit Rate@5</div>
            </div>
            <div class="stat-card accent-purple">
                <div class="stat-value">{r.avg_recall_at_1:.2%}</div>
                <div class="stat-label">Recall@1</div>
            </div>
        </div>

        <div class="metrics-table">
            <h3>核心指标汇总</h3>
            <table>
                <thead><tr><th>指标</th><th>K=1</th><th>K=3</th><th>K=5</th><th>K=10</th></tr></thead>
                <tbody>
                    <tr><td>Recall@K</td><td>{r.avg_recall_at_1:.4f}</td><td>{r.avg_recall_at_3:.4f}</td><td>{r.avg_recall_at_5:.4f}</td><td>{r.avg_recall_at_10:.4f}</td></tr>
                    <tr><td>Hit Rate@K</td><td>{r.hit_rate_at_1:.4f}</td><td>{r.hit_rate_at_3:.4f}</td><td>{r.hit_rate_at_5:.4f}</td><td>{r.hit_rate_at_10:.4f}</td></tr>
                </tbody>
            </table>
            <table style="margin-top:16px">
                <thead><tr><th>指标</th><th>值</th></tr></thead>
                <tbody>
                    <tr><td>MRR</td><td>{r.avg_mrr:.4f}</td></tr>
                    <tr><td>Avg Embed (ms)</td><td>{r.avg_embed_ms:.1f}</td></tr>
                    <tr><td>Avg Vector Search (ms)</td><td>{r.avg_vector_search_ms:.1f}</td></tr>
                    <tr><td>Avg Rerank (ms)</td><td>{r.avg_rerank_ms:.1f}</td></tr>
                </tbody>
            </table>
        </div>

        <div class="charts-row">
            <div class="chart-box">
                <h3>Recall@K 随 K 变化</h3>
                <canvas id="recallChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>Hit Rate@K</h3>
                <canvas id="hitRateChart"></canvas>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-box">
                <h3>不同难度 Recall@K 对比</h3>
                <canvas id="recallDifficultyChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>不同难度 MRR 对比</h3>
                <canvas id="mrrDifficultyChart"></canvas>
            </div>
        </div>

        <h3>逐题检索结果 (Top 20)</h3>
        <div class="table-scroll">
            <table class="detail-table">
                <thead><tr><th>ID</th><th>问题</th><th>类别</th><th>难度</th><th>R@1</th><th>R@5</th><th>R@10</th><th>MRR</th><th>Hit@5</th><th>Top-1 Score</th></tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </div>
    </div>

    <script>
    new Chart(document.getElementById('recallChart'), {{
        type: 'line',
        data: {{
            labels: ['K=1','K=3','K=5','K=10'],
            datasets: [{{
                label: 'Recall@K',
                data: [{r.avg_recall_at_1},{r.avg_recall_at_3},{r.avg_recall_at_5},{r.avg_recall_at_10}],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.1)',
                fill: true,
                tension: 0.3
            }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
    }});
    new Chart(document.getElementById('hitRateChart'), {{
        type: 'bar',
        data: {{
            labels: ['K=1','K=3','K=5','K=10'],
            datasets: [{{
                label: 'Hit Rate',
                data: [{r.hit_rate_at_1},{r.hit_rate_at_3},{r.hit_rate_at_5},{r.hit_rate_at_10}],
                backgroundColor: '#10b981'
            }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
    }});
    _chartRecallDifficulty({json.dumps(r.recall_by_difficulty, ensure_ascii=False)});
    _chartMrrDifficulty({json.dumps(r.mrr_by_difficulty, ensure_ascii=False)});
    </script>
    """


def _build_qa_section(r: QAEvalResult) -> str:
    detail_rows = ""
    for q in r.questions[:50]:
        detail_rows += f"""
        <tr>
            <td>{q.question_id}</td>
            <td class="q-text" title="{_esc(q.question[:80])}">{_esc(q.question[:50])}...</td>
            <td><span class="badge badge-{q.difficulty}">{q.difficulty}</span></td>
            <td>{q.bleu_1:.3f}</td>
            <td>{q.bleu_4:.3f}</td>
            <td>{q.rouge_l_f:.3f}</td>
            <td>{q.faithfulness_score}</td>
            <td>{q.overall_score:.1f}</td>
        </tr>
        """

    return f"""
    <div class="section">
        <h2>3. 问答效果评估 (5.4)</h2>

        <div class="stats-grid">
            <div class="stat-card accent-blue">
                <div class="stat-value">{r.avg_bleu_1:.1%}</div>
                <div class="stat-label">Avg BLEU-1</div>
            </div>
            <div class="stat-card accent-green">
                <div class="stat-value">{r.avg_rouge_l_f:.1%}</div>
                <div class="stat-label">Avg ROUGE-L</div>
            </div>
            <div class="stat-card accent-orange">
                <div class="stat-value">{r.avg_faithfulness:.1f}/5</div>
                <div class="stat-label">Avg Faithfulness</div>
            </div>
            <div class="stat-card accent-purple">
                <div class="stat-value">{r.avg_overall:.1f}/5</div>
                <div class="stat-label">Avg Human Score</div>
            </div>
        </div>

        <div class="metrics-table">
            <h3>核心指标汇总</h3>
            <table>
                <thead><tr><th>指标</th><th>值</th></tr></thead>
                <tbody>
                    <tr><td>BLEU-1</td><td>{r.avg_bleu_1:.4f}</td></tr>
                    <tr><td>BLEU-2</td><td>{r.avg_bleu_2:.4f}</td></tr>
                    <tr><td>BLEU-4</td><td>{r.avg_bleu_4:.4f}</td></tr>
                    <tr><td>ROUGE-1 F</td><td>{r.avg_rouge_1_f:.4f}</td></tr>
                    <tr><td>ROUGE-2 F</td><td>{r.avg_rouge_2_f:.4f}</td></tr>
                    <tr><td>ROUGE-L F</td><td>{r.avg_rouge_l_f:.4f}</td></tr>
                    <tr><td>Faithfulness</td><td>{r.avg_faithfulness:.2f} / 5</td></tr>
                    <tr><td>Relevance</td><td>{r.avg_relevance:.2f} / 5</td></tr>
                    <tr><td>Completeness</td><td>{r.avg_completeness:.2f} / 5</td></tr>
                    <tr><td>Accuracy</td><td>{r.avg_accuracy:.2f} / 5</td></tr>
                    <tr><td>Overall Human Score</td><td>{r.avg_overall:.2f} / 5</td></tr>
                </tbody>
            </table>
        </div>

        <div class="charts-row">
            <div class="chart-box">
                <h3>BLEU/ROUGE 对比</h3>
                <canvas id="bleuRougeChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>人工评分分布</h3>
                <canvas id="humanChart"></canvas>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-box">
                <h3>Faithfulness 分数分布</h3>
                <canvas id="faithfulnessDistChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>不同难度 BLEU-4 对比</h3>
                <canvas id="bleuDifficultyChart"></canvas>
            </div>
        </div>

        <h3>逐题问答结果 (Top 20)</h3>
        <div class="table-scroll">
            <table class="detail-table">
                <thead><tr><th>ID</th><th>问题</th><th>难度</th><th>BLEU-1</th><th>BLEU-4</th><th>ROUGE-L</th><th>Faith.</th><th>Overall</th></tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </div>
    </div>

    <script>
    new Chart(document.getElementById('bleuRougeChart'), {{
        type: 'bar',
        data: {{
            labels: ['BLEU-1','BLEU-2','BLEU-4','ROUGE-1','ROUGE-2','ROUGE-L'],
            datasets: [{{
                label: 'Score',
                data: [{r.avg_bleu_1},{r.avg_bleu_2},{r.avg_bleu_4},{r.avg_rouge_1_f},{r.avg_rouge_2_f},{r.avg_rouge_l_f}],
                backgroundColor: ['#3b82f6','#60a5fa','#93c5fd','#10b981','#34d399','#6ee7b7']
            }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0, max: {_ceil(max(r.avg_bleu_1, r.avg_rouge_l_f) + 0.1)} }} }} }}
    }});
    new Chart(document.getElementById('humanChart'), {{
        type: 'bar',
        data: {{
            labels: ['Faithfulness','Relevance','Completeness','Accuracy','Overall'],
            datasets: [{{
                label: 'Avg Score',
                data: [{r.avg_faithfulness},{r.avg_relevance},{r.avg_completeness},{r.avg_accuracy},{r.avg_overall}],
                backgroundColor: ['#8b5cf6','#3b82f6','#10b981','#f59e0b','#ef4444']
            }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 5 }} }} }}
    }});
    _chartFaithfulnessDist({json.dumps([sum(1 for q in r.questions if q.faithfulness_score == s) for s in range(1, 6)])});
    _chartBleuDifficulty({json.dumps(r.bleu_by_difficulty, ensure_ascii=False)});
    </script>
    """


def _build_analysis_section(ret: RetrievalEvalResult, qa: QAEvalResult) -> str:
    total_q = len(ret.questions)
    good_retrieval = sum(1 for q in ret.questions if q.hit_at_5)
    good_qa = sum(1 for q in qa.questions if q.overall_score >= 3.5)

    # Per-question alignment analysis
    alignment = 0
    for rq, aq in zip(ret.questions, qa.questions):
        if rq.hit_at_5 and aq.overall_score >= 3.5:
            alignment += 1
        elif not rq.hit_at_5 and aq.overall_score >= 3.5:
            alignment += 1  # Good QA despite poor retrieval = LLM compensated
        elif rq.hit_at_5 and aq.overall_score < 3.5:
            alignment += 1  # Poor QA despite good retrieval = generation issue

    # Generate recommendations
    recommendations: list[str] = []
    if ret.avg_mrr < 0.5:
        recommendations.append("MRR 较低，建议优化检索策略（如增加 rerank 候选数或引入 hybrid search）")
    if ret.avg_recall_at_1 < 0.5:
        recommendations.append("Top-1 召回率偏低，建议优化 embedding 模型或增加 top-k 候选数")
    if qa.avg_faithfulness < 3.5:
        recommendations.append("Faithfulness 偏低，建议优化 System Prompt 加强对上下文的约束")
    if qa.avg_accuracy < 3.5:
        recommendations.append("准确性偏低，建议检查 LLM 是否对 retrieved context 利用不足")
    if not recommendations:
        recommendations.append("系统整体表现良好，可在特定类别上进一步微调")

    rec_items = "".join(f"<li>{r}</li>" for r in recommendations)

    return f"""
    <div class="section">
        <h2>4. 综合分析</h2>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{good_retrieval}/{total_q}</div>
                <div class="stat-label">检索命中 (Hit@5) 问题数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{good_qa}/{total_q}</div>
                <div class="stat-label">问答评分 ≥3.5 问题数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{qa.avg_overall:.1f}</div>
                <div class="stat-label">系统整体评分</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{ret.hit_rate_at_5:.0%}</div>
                <div class="stat-label">端到端正确率</div>
            </div>
        </div>

        <div class="analysis-box">
            <h3>关键发现</h3>
            <ul>
                <li>检索 MRR 为 {ret.avg_mrr:.3f}，表明检索系统平均在第 {_rank_text(ret.avg_mrr)} 个位置找到首个相关文档</li>
                <li>Recall@5 为 {ret.avg_recall_at_5:.1%}，即在前 5 个结果中能覆盖 {ret.avg_recall_at_5:.0%} 的相关 chunk</li>
                <li>平均 ROUGE-L 为 {qa.avg_rouge_l_f:.1%}，反映生成回答与参考答案的文本相似度</li>
                <li>Faithfulness 评分 {qa.avg_faithfulness:.1f}/5，表明生成回答忠实度{"较高" if qa.avg_faithfulness >= 3.5 else "可优化"}</li>
                <li>人工模拟综合评分 {qa.avg_overall:.1f}/5，反映回答的整体质量</li>
            </ul>
        </div>

        <div class="analysis-box">
            <h3>改进建议</h3>
            <ol>{rec_items}</ol>
        </div>

        <div class="analysis-box">
            <h3>检索延迟分析</h3>
            <table>
                <thead><tr><th>阶段</th><th>平均耗时 (ms)</th><th>占比</th></tr></thead>
                <tbody>
                    <tr><td>Embedding</td><td>{ret.avg_embed_ms:.1f}</td><td>{_pct(ret.avg_embed_ms, ret.avg_embed_ms + ret.avg_vector_search_ms + ret.avg_rerank_ms)}%</td></tr>
                    <tr><td>Vector Search</td><td>{ret.avg_vector_search_ms:.1f}</td><td>{_pct(ret.avg_vector_search_ms, ret.avg_embed_ms + ret.avg_vector_search_ms + ret.avg_rerank_ms)}%</td></tr>
                    <tr><td>Rerank</td><td>{ret.avg_rerank_ms:.1f}</td><td>{_pct(ret.avg_rerank_ms, ret.avg_embed_ms + ret.avg_vector_search_ms + ret.avg_rerank_ms)}%</td></tr>
                    <tr><td>Total</td><td>{ret.avg_embed_ms + ret.avg_vector_search_ms + ret.avg_rerank_ms:.1f}</td><td>100%</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    """


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _ceil(v: float) -> float:
    import math

    return math.ceil(v * 10) / 10


def _rank_text(mrr: float) -> str:
    if mrr <= 0:
        return "N/A"
    rank = 1.0 / mrr
    return f"{rank:.1f}"


def _pct(part: float, total: float) -> str:
    if total <= 0:
        return "0"
    return f"{part / total * 100:.0f}"


_WRAPPER = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ text-align: center; padding: 40px 0 20px; }}
.header h1 {{ font-size: 2rem; color: #0f172a; }}
.header .subtitle {{ color: #64748b; margin-top: 8px; }}
.section {{ background: #fff; border-radius: 12px; padding: 32px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.section h2 {{ font-size: 1.4rem; color: #1e40af; margin-bottom: 20px; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.stat-card {{ background: #f1f5f9; border-radius: 8px; padding: 20px; text-align: center; }}
.stat-card.accent-blue {{ border-left: 4px solid #3b82f6; }}
.stat-card.accent-green {{ border-left: 4px solid #10b981; }}
.stat-card.accent-orange {{ border-left: 4px solid #f59e0b; }}
.stat-card.accent-purple {{ border-left: 4px solid #8b5cf6; }}
.stat-value {{ font-size: 1.8rem; font-weight: 700; color: #0f172a; }}
.stat-label {{ font-size: 0.85rem; color: #64748b; margin-top: 4px; }}
.charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
.chart-box {{ background: #f8fafc; border-radius: 8px; padding: 16px; }}
.chart-box h3 {{ font-size: 1rem; color: #334155; margin-bottom: 12px; }}
.chart-box canvas {{ max-height: 280px; }}
.tables-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.metrics-table {{ margin-bottom: 24px; }}
table {{ width: 100%; border-collapse: collapse; }}
thead th {{ background: #f1f5f9; padding: 10px 12px; text-align: left; font-weight: 600; font-size: 0.85rem; color: #475569; border-bottom: 2px solid #e2e8f0; }}
tbody td {{ padding: 8px 12px; font-size: 0.9rem; border-bottom: 1px solid #f1f5f9; }}
tbody tr:hover {{ background: #f8fafc; }}
.q-text {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
.badge-easy {{ background: #d1fae5; color: #065f46; }}
.badge-medium {{ background: #fef3c7; color: #92400e; }}
.badge-hard {{ background: #fee2e2; color: #991b1b; }}
.table-scroll {{ overflow-x: auto; }}
.detail-table {{ font-size: 0.8rem; }}
.analysis-box {{ background: #f8fafc; border-radius: 8px; padding: 20px; margin-bottom: 16px; }}
.analysis-box h3 {{ font-size: 1rem; color: #334155; margin-bottom: 12px; }}
.analysis-box ul, .analysis-box ol {{ padding-left: 20px; }}
.analysis-box li {{ margin-bottom: 6px; font-size: 0.9rem; }}
.footer {{ text-align: center; padding: 20px; color: #94a3b8; font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="container">
{body}
<div class="footer">Generated at {generated_at} · RAG System Evaluation</div>
</div>
<script>
function _chartRecallDifficulty(data) {{
    var keys = Object.keys(data);
    if (!keys.length) return;
    var datasets = [];
    var colors = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6'];
    var kValues = [1,3,5,10];
    keys.forEach(function(key, i) {{
        datasets.push({{
            label: key,
            data: kValues.map(function(k) {{ return data[key][k] || 0; }}),
            borderColor: colors[i % colors.length],
            fill: false,
            tension: 0.3
        }});
    }});
    var canvas = document.getElementById('recallDifficultyChart');
    if (canvas) {{
        new Chart(canvas, {{ type: 'line', data: {{ labels: ['K=1','K=3','K=5','K=10'], datasets: datasets }}, options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }} }});
    }}
}}
function _chartMrrDifficulty(data) {{
    var keys = Object.keys(data);
    if (!keys.length) return;
    var canvas = document.getElementById('mrrDifficultyChart');
    if (canvas) {{
        new Chart(canvas, {{
            type: 'bar',
            data: {{
                labels: keys,
                datasets: [{{ label: 'MRR', data: keys.map(function(k) {{ return data[k]; }}), backgroundColor: '#8b5cf6' }}]
            }},
            options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
        }});
    }}
}}
function _chartFaithfulnessDist(data) {{
    var canvas = document.getElementById('faithfulnessDistChart');
    if (canvas) {{
        new Chart(canvas, {{
            type: 'bar',
            data: {{
                labels: ['1分','2分','3分','4分','5分'],
                datasets: [{{
                    label: '问题数',
                    data: data,
                    backgroundColor: ['#ef4444','#f97316','#f59e0b','#10b981','#3b82f6']
                }}]
            }},
            options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
        }});
    }}
}}
function _chartBleuDifficulty(data) {{
    var keys = Object.keys(data);
    if (!keys.length) return;
    var canvas = document.getElementById('bleuDifficultyChart');
    if (canvas) {{
        var datasets = [];
        var colors = ['#3b82f6','#10b981','#f59e0b'];
        ['bleu_1','bleu_2','bleu_4'].forEach(function(metric, i) {{
            datasets.push({{
                label: metric.toUpperCase().replace('_','-'),
                data: keys.map(function(k) {{ return data[k][metric] || 0; }}),
                backgroundColor: colors[i]
            }});
        }});
        new Chart(canvas, {{ type: 'bar', data: {{ labels: keys, datasets: datasets }}, options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }} }});
    }}
}}
</script>
</body>
</html>"""
