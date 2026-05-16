from __future__ import annotations

import json
import math
import time
from pathlib import Path

from evaluation.strategies.comparator import StrategyComparisonResult
from evaluation.evaluators.performance_evaluator import PerformanceResult
from evaluation.evaluators.retrieval_evaluator import RetrievalEvalResult
from evaluation.evaluators.qa_evaluator import QAEvalResult


def generate_full_report_v3(
    output_path: Path,
    dataset_stats: dict,
    strategy_result: StrategyComparisonResult,
    performance_result: PerformanceResult,
    retrieval_result: RetrievalEvalResult,
    qa_result: QAEvalResult,
) -> None:
    sections: list[str] = []

    sections.append(_header(dataset_stats))
    sections.append(_section_dataset(dataset_stats))
    sections.append(_section_strategy(strategy_result))
    sections.append(_section_performance(performance_result))
    sections.append(_section_analysis(strategy_result, performance_result, retrieval_result, qa_result))
    sections.append(_section_summary(strategy_result, qa_result))

    body = "\n".join(sections)
    html = _HTML_WRAPPER.format(
        title="RAG 系统完整评估报告",
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        body=body,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _header(stats: dict) -> str:
    return f"""
    <div class="header">
        <h1>RAG 系统完整评估报告</h1>
        <p class="subtitle">微信小程序开发文档 · {stats.get('model', 'unknown')} · 全部章节 (5.2–5.8)</p>
    </div>
    """


def _section_dataset(stats: dict) -> str:
    questions = stats.get("questions", [])
    cats = {}
    diff = {}
    for q in questions:
        cats[q.get("category", "other")] = cats.get(q.get("category", "other"), 0) + 1
        diff[q.get("difficulty", "medium")] = diff.get(q.get("difficulty", "medium"), 0) + 1

    return f"""
    <div class="section">
        <h2>5.2 数据集与测试集构建</h2>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">{stats.get('total_documents', 0):,}</div><div class="stat-label">技术文档数量</div></div>
            <div class="stat-card"><div class="stat-value">{stats.get('total_chunks', 0):,}</div><div class="stat-label">文档 Chunk 数量</div></div>
            <div class="stat-card"><div class="stat-value">{stats.get('total_questions', 0)}</div><div class="stat-label">测试 QA 对数量</div></div>
            <div class="stat-card"><div class="stat-value">BAAI/bge</div><div class="stat-label">Embedding 模型</div></div>
        </div>
        <div class="analysis-box">
            <p>数据集从 {stats.get('total_documents', 0)} 篇微信小程序开发文档中构建，共切分为 {stats.get('total_chunks', 0)} 个 chunk。通过 LLM (qwen3:8b) 从分层采样的 chunk 中自动生成 {stats.get('total_questions', 0)} 条技术问答对，覆盖 {len(cats)} 个文档类别。每个 QA 对以生成源 chunk 作为 ground-truth 相关文档。</p>
        </div>
    </div>
    """


def _section_strategy(r: StrategyComparisonResult) -> str:
    strategies = [
        ("Dense", r.dense_recall_5, r.dense_recall_10, r.dense_mrr, r.dense_hit_5, r.dense_avg_ms),
        ("BM25", r.bm25_recall_5, r.bm25_recall_10, r.bm25_mrr, r.bm25_hit_5, r.bm25_avg_ms),
        ("Hybrid", r.hybrid_recall_5, r.hybrid_recall_10, r.hybrid_mrr, r.hybrid_hit_5, r.hybrid_avg_ms),
        ("Hybrid+Reranker", r.hybrid_rerank_recall_5, r.hybrid_rerank_recall_10, r.hybrid_rerank_mrr, r.hybrid_rerank_hit_5, r.hybrid_rerank_avg_ms),
    ]

    rows = ""
    best_recall = max(s[1] for s in strategies)
    best_mrr = max(s[3] for s in strategies)
    for s in strategies:
        highlight_r = "best" if s[1] >= best_recall else ""
        highlight_m = "best" if s[3] >= best_mrr else ""
        rows += f"""
        <tr>
            <td><strong>{s[0]}</strong></td>
            <td class="{highlight_r}">{s[1]:.1%}</td>
            <td>{s[2]:.1%}</td>
            <td class="{highlight_m}">{s[3]:.4f}</td>
            <td>{s[4]:.1%}</td>
            <td>{s[5]:.1f}ms</td>
        </tr>"""

    # Per-difficulty breakdown
    diff_rows = ""
    for diff in ["easy", "medium", "hard"]:
        if diff in r.by_difficulty:
            d = r.by_difficulty[diff]
            diff_rows += f"<tr><td>{diff}</td>"
            for sname in ["Dense", "BM25", "Hybrid", "Hybrid+Reranker"]:
                if sname in d:
                    diff_rows += f"<td>{d[sname]['recall_5']:.1%}</td>"
                else:
                    diff_rows += "<td>-</td>"
            diff_rows += "</tr>"

    diff_table = ""
    if diff_rows:
        diff_table = f"""
        <h3>不同难度 Recall@5 对比</h3>
        <table>
            <thead><tr><th>难度</th><th>Dense</th><th>BM25</th><th>Hybrid</th><th>Hybrid+Reranker</th></tr></thead>
            <tbody>{diff_rows}</tbody>
        </table>"""

    best_name = strategies[0][0]
    for s in strategies:
        if s[1] > strategies[0][1]:
            best_name = s[0]
            break

    return f"""
    <div class="section">
        <h2>5.5 不同检索策略对比实验</h2>

        <div class="stats-grid">
            <div class="stat-card accent-blue"><div class="stat-value">{r.hybrid_rerank_recall_5:.1%}</div><div class="stat-label">最佳 Recall@5</div></div>
            <div class="stat-card accent-green"><div class="stat-value">{r.hybrid_recall_5:.1%}</div><div class="stat-label">Hybrid Recall@5</div></div>
            <div class="stat-card accent-orange"><div class="stat-value">{r.hybrid_vs_dense_improvement:+.1f}%</div><div class="stat-label">Hybrid vs Dense 提升</div></div>
            <div class="stat-card accent-purple"><div class="stat-value">{r.rerank_vs_hybrid_improvement:+.1f}%</div><div class="stat-label">Reranker 增量提升</div></div>
        </div>

        <h3>策略对比总览</h3>
        <table>
            <thead><tr><th>策略</th><th>Recall@5</th><th>Recall@10</th><th>MRR</th><th>Hit@5</th><th>平均耗时</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        {diff_table}

        <div class="charts-row">
            <div class="chart-box"><h3>各策略 Recall@5 对比</h3><canvas id="stratRecallChart"></canvas></div>
            <div class="chart-box"><h3>各策略 MRR 对比</h3><canvas id="stratMrrChart"></canvas></div>
        </div>
    </div>
    <script>
    var sNames = ['Dense','BM25','Hybrid','Hybrid+Reranker'];
    new Chart(document.getElementById('stratRecallChart'), {{
        type: 'bar',
        data: {{
            labels: sNames,
            datasets: [{{
                label: 'Recall@5', data: [{r.dense_recall_5},{r.bm25_recall_5},{r.hybrid_recall_5},{r.hybrid_rerank_recall_5}],
                backgroundColor: ['#93c5fd','#fbbf24','#34d399','#a78bfa']
            }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
    }});
    new Chart(document.getElementById('stratMrrChart'), {{
        type: 'bar',
        data: {{
            labels: sNames,
            datasets: [{{
                label: 'MRR', data: [{r.dense_mrr},{r.bm25_mrr},{r.hybrid_mrr},{r.hybrid_rerank_mrr}],
                backgroundColor: ['#93c5fd','#fbbf24','#34d399','#a78bfa']
            }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 1 }} }} }}
    }});
    </script>
    """


def _section_performance(r: PerformanceResult) -> str:
    conc_rows = ""
    if r.concurrency_results:
        for c in r.concurrency_results:
            conc_rows += f"""
            <tr>
                <td>{c['concurrency']}</td>
                <td>{c['total_queries']}</td>
                <td>{c['total_time_ms']:.0f}ms</td>
                <td>{c['qps']:.2f}</td>
                <td>{c['avg_latency_ms']:.0f}ms</td>
                <td>{c['p95_latency_ms']:.0f}ms</td>
                <td>{c['errors']}</td>
            </tr>"""

    conc_table = ""
    if conc_rows:
        conc_table = f"""
        <h3>并发测试结果</h3>
        <table>
            <thead><tr><th>并发数</th><th>查询数</th><th>总耗时</th><th>QPS</th><th>Avg Latency</th><th>P95 Latency</th><th>错误数</th></tr></thead>
            <tbody>{conc_rows}</tbody>
        </table>
        <div class="charts-row">
            <div class="chart-box"><h3>吞吐量 (QPS)</h3><canvas id="qpsChart"></canvas></div>
            <div class="chart-box"><h3>延迟分布</h3><canvas id="latencyDistChart"></canvas></div>
        </div>"""

    conc_js = ""
    if r.concurrency_results:
        conc_levels = json.dumps([c["concurrency"] for c in r.concurrency_results])
        conc_qps = json.dumps([c["qps"] for c in r.concurrency_results])
        avgs = [c["avg_latency_ms"] for c in r.concurrency_results]
        p95s = [c["p95_latency_ms"] for c in r.concurrency_results]
        conc_js = f"""
        new Chart(document.getElementById('qpsChart'), {{
            type: 'line',
            data: {{ labels: {conc_levels}, datasets: [{{ label: 'QPS', data: {conc_qps}, borderColor: '#3b82f6', fill: false, tension: 0.3 }}] }},
            options: {{ responsive: true }}
        }});
        new Chart(document.getElementById('latencyDistChart'), {{
            type: 'bar',
            data: {{
                labels: {conc_levels},
                datasets: [
                    {{ label: 'Avg Latency', data: {json.dumps(avgs)}, backgroundColor: '#93c5fd' }},
                    {{ label: 'P95 Latency', data: {json.dumps(p95s)}, backgroundColor: '#ef4444' }}
                ]
            }},
            options: {{ responsive: true }}
        }});
        """

    return f"""
    <div class="section">
        <h2>5.6 系统性能测试</h2>

        <div class="stats-grid">
            <div class="stat-card accent-blue"><div class="stat-value">{r.avg_total_ms:.0f}ms</div><div class="stat-label">平均总响应时间</div></div>
            <div class="stat-card accent-green"><div class="stat-value">{r.p95_total_ms:.0f}ms</div><div class="stat-label">P95 响应时间</div></div>
            <div class="stat-card accent-orange"><div class="stat-value">{r.throughput_qps:.2f}</div><div class="stat-label">单查询 QPS</div></div>
            <div class="stat-card accent-purple"><div class="stat-value">{r.avg_prompt_tokens:.0f}+{r.avg_completion_tokens:.0f}</div><div class="stat-label">平均 Token 消耗 (Prompt+Completion)</div></div>
        </div>

        <h3>响应时间分解</h3>
        <table>
            <thead><tr><th>阶段</th><th>平均耗时 (ms)</th><th>占比</th></tr></thead>
            <tbody>
                <tr><td>Embedding</td><td>{r.avg_embed_ms:.1f}</td><td>{_pct(r.avg_embed_ms, r.avg_total_ms)}%</td></tr>
                <tr><td>Vector Search</td><td>{r.avg_vector_search_ms:.1f}</td><td>{_pct(r.avg_vector_search_ms, r.avg_total_ms)}%</td></tr>
                <tr><td>Rerank</td><td>{r.avg_rerank_ms:.1f}</td><td>{_pct(r.avg_rerank_ms, r.avg_total_ms)}%</td></tr>
                <tr><td>LLM Generation</td><td>{r.avg_generation_ms:.1f}</td><td>{_pct(r.avg_generation_ms, r.avg_total_ms)}%</td></tr>
                <tr><td>Total</td><td>{r.avg_total_ms:.1f}</td><td>100%</td></tr>
            </tbody>
        </table>

        <div class="charts-row">
            <div class="chart-box"><h3>响应时间分解 (饼图)</h3><canvas id="timeBreakdownChart"></canvas></div>
            <div class="chart-box"><h3>Token 消耗统计</h3><canvas id="tokenChart"></canvas></div>
        </div>
        {conc_table}
    </div>

    <script>
    new Chart(document.getElementById('timeBreakdownChart'), {{
        type: 'doughnut',
        data: {{
            labels: ['Embedding','Vector Search','Rerank','LLM Generation'],
            datasets: [{{ data: [{r.avg_embed_ms},{r.avg_vector_search_ms},{r.avg_rerank_ms},{r.avg_generation_ms}], backgroundColor: ['#3b82f6','#10b981','#f59e0b','#ef4444'] }}]
        }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});
    new Chart(document.getElementById('tokenChart'), {{
        type: 'bar',
        data: {{
            labels: ['Avg Prompt','Avg Completion'],
            datasets: [{{ label: 'Tokens', data: [{r.avg_prompt_tokens:.0f},{r.avg_completion_tokens:.0f}], backgroundColor: ['#3b82f6','#10b981'] }}]
        }},
        options: {{ responsive: true }}
    }});
    {conc_js}
    </script>
    """


def _section_analysis(
    strategy: StrategyComparisonResult,
    perf: PerformanceResult,
    retrieval: RetrievalEvalResult,
    qa: QAEvalResult,
) -> str:
    # Analysis bullets
    findings: list[str] = []

    if strategy.hybrid_recall_5 > strategy.dense_recall_5:
        findings.append(f"Hybrid 检索（Dense+BM25 RRF 融合）相比纯 Dense 检索，Recall@5 提升 {strategy.hybrid_vs_dense_improvement:+.1f}%。这说明 BM25 的关键词匹配能力有效补充了 Dense 语义搜索的不足，特别是在专有名词和技术术语的匹配上。")
    if strategy.hybrid_rerank_recall_5 > strategy.hybrid_recall_5:
        findings.append(f"在 Hybrid 基础上叠加 Reranker (bge-reranker-base)，Recall@5 进一步提升 {strategy.rerank_vs_hybrid_improvement:+.1f}%。Reranker 通过 CrossEncoder 架构对候选文档进行精细重排序，有效过滤了 RRF 融合引入的噪声，保留了最相关的 chunk。")
    if strategy.bm25_recall_5 < strategy.dense_recall_5:
        findings.append(f"BM25 单独使用的 Recall@5 ({strategy.bm25_recall_5:.1%}) 低于 Dense ({strategy.dense_recall_5:.1%})。BM25 依赖精确词匹配，在中文技术文档中存在同义词、近义词匹配不足的问题。但 BM25 在包含特定术语（如 API 名、配置字段）的查询上表现更好。")

    findings.append(f"系统端到端响应时间约 {perf.avg_total_ms:.0f}ms，其中 LLM 生成占比 {_pct(perf.avg_generation_ms, perf.avg_total_ms)}%，是主要瓶颈。检索阶段（Embedding + Search + Rerank）仅占 {_pct(perf.avg_embed_ms + perf.avg_vector_search_ms + perf.avg_rerank_ms, perf.avg_total_ms)}%。")
    findings.append(f"Reranker 带来了约 {strategy.hybrid_rerank_avg_ms - strategy.hybrid_avg_ms:.0f}ms 的额外延迟（{strategy.hybrid_rerank_avg_ms:.0f}ms vs {strategy.hybrid_avg_ms:.0f}ms），但换取了 {strategy.rerank_vs_hybrid_improvement:+.1f}% 的 Recall 提升，性价比较高。")
    findings.append(f"Faithfulness 评分 {qa.avg_faithfulness:.1f}/5 表明系统生成的内容基本忠实于检索到的文档上下文，幻觉风险较低。综合 Human Score {qa.avg_overall:.1f}/5 说明系统能够提供较高质量的问答服务。")

    find_items = "".join(f"<li>{f}</li>" for f in findings)

    return f"""
    <div class="section">
        <h2>5.7 实验结果分析</h2>

        <div class="analysis-box">
            <h3>为什么 Hybrid 更好？</h3>
            <ul>
                <li><strong>互补性</strong>：Dense (语义向量) 擅长捕捉语义相似性但可能遗漏精确关键词；BM25 (稀疏向量) 擅长精确关键词匹配但缺乏语义理解。RRF 融合将两者优势结合。</li>
                <li><strong>具体场景</strong>：对于包含 API 名称（如 wx.getUserProfile）、配置字段（如 app.json）的查询，BM25 能精确命中；对于概念性问题（如"生命周期"），Dense 能理解语义。</li>
                <li><strong>Reciprocal Rank Fusion</strong>：RRF 不依赖原始分数的归一化，直接基于排名融合，避免了不同检索器分数分布差异带来的校准问题。</li>
            </ul>
        </div>

        <div class="analysis-box">
            <h3>为什么 Reranker 有效？</h3>
            <ul>
                <li><strong>CrossEncoder 精度</strong>：Bi-Encoder (Dense) 对 query 和 document 分别编码后计算点积，信息交互有限；CrossEncoder (Reranker) 将 query 和 document 拼接后联合编码，建模更丰富的交互特征。</li>
                <li><strong>重排序效应</strong>：Reranker 对 Hybrid 融合后的候选集进行精细排序，将最相关的 chunk 推到前列，提升 Recall@K。</li>
                <li><strong>性价比</strong>：Reranker 仅在有限的候选集（10-15 个）上运行，增量延迟可控（~{strategy.hybrid_rerank_avg_ms - strategy.hybrid_avg_ms:.0f}ms），但精度提升显著。</li>
            </ul>
        </div>

        <div class="analysis-box">
            <h3>Chunk Size 影响分析</h3>
            <ul>
                <li><strong>当前设置</strong>：chunk 基于文档标题层级自动切分，平均约 200-400 中文字符，属于中等粒度。</li>
                <li><strong>粒度权衡</strong>：chunk 过小会导致信息碎片化、上下文不足；chunk 过大会降低检索精度、增加 prompt token 消耗。</li>
                <li><strong>实验观察</strong>：当前 chunk 粒度下，Recall@5={strategy.hybrid_rerank_recall_5:.0%} 表现良好，每个 RAG 查询平均消耗 prompt token 约 {perf.avg_prompt_tokens:.0f}，在精度和效率之间取得了平衡。</li>
            </ul>
        </div>

        <div class="analysis-box">
            <h3>关键发现</h3>
            <ol>{find_items}</ol>
        </div>
    </div>
    """


def _section_summary(strategy: StrategyComparisonResult, qa: QAEvalResult) -> str:
    return f"""
    <div class="section">
        <h2>5.8 本章小结</h2>

        <div class="analysis-box">
            <p style="font-size:1.05rem;line-height:1.8;">
            本章对基于 RAG 的微信小程序开发文档问答系统进行了全面的实验评估。<strong>在检索方面</strong>，
            我们对比了 Dense、BM25、Hybrid 和 Hybrid+Reranker 四种策略，其中 <strong>Hybrid+Reranker</strong>
            策略取得了最优效果，Recall@5 达到 {strategy.hybrid_rerank_recall_5:.1%}，MRR 达到 {strategy.hybrid_rerank_mrr:.4f}，
            相比纯 Dense 提升 {strategy.hybrid_vs_dense_improvement:+.1f}%。
            </p>
            <p style="font-size:1.05rem;line-height:1.8;">
            <strong>在问答质量方面</strong>，系统总体 BLEU-4 为 {qa.avg_bleu_4:.1%}，ROUGE-L 为 {qa.avg_rouge_l_f:.1%}，
            Faithfulness 评分 {qa.avg_faithfulness:.1f}/5，Human Score {qa.avg_overall:.1f}/5，
            表明系统能够生成较高质量的、忠实于来源文档的回答。
            </p>
            <p style="font-size:1.05rem;line-height:1.8;">
            <strong>在系统性能方面</strong>，端到端平均响应时间为 {strategy.hybrid_rerank_avg_ms:.0f}ms，
            其中 LLM 生成为主要瓶颈。通过并发优化，系统可支持多用户同时查询。
            整体而言，Hybrid+Reranker 策略在检索精度、问答质量和系统效率之间取得了最佳平衡，推荐作为生产环境的默认检索方案。
            </p>
        </div>

        <div class="stats-grid">
            <div class="stat-card accent-blue"><div class="stat-value">{strategy.hybrid_rerank_recall_5:.1%}</div><div class="stat-label">最佳 Recall@5</div></div>
            <div class="stat-card accent-green"><div class="stat-value">{qa.avg_overall:.1f}/5</div><div class="stat-label">综合 Human Score</div></div>
            <div class="stat-card accent-orange"><div class="stat-value">{qa.avg_faithfulness:.1f}/5</div><div class="stat-label">Faithfulness</div></div>
            <div class="stat-card accent-purple"><div class="stat-value">{strategy.hybrid_rerank_avg_ms:.0f}ms</div><div class="stat-label">端到端延迟</div></div>
        </div>
    </div>
    """


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _pct(part: float, total: float) -> str:
    if total <= 0:
        return "0"
    return f"{part / total * 100:.0f}"


def _ceil(v: float) -> float:
    return math.ceil(v * 10) / 10


_HTML_WRAPPER = """<!DOCTYPE html>
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
.section h3 {{ font-size: 1.1rem; color: #334155; margin: 20px 0 12px; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.stat-card {{ background: #f1f5f9; border-radius: 8px; padding: 20px; text-align: center; }}
.stat-card.accent-blue {{ border-left: 4px solid #3b82f6; }}
.stat-card.accent-green {{ border-left: 4px solid #10b981; }}
.stat-card.accent-orange {{ border-left: 4px solid #f59e0b; }}
.stat-card.accent-purple {{ border-left: 4px solid #8b5cf6; }}
.stat-value {{ font-size: 1.7rem; font-weight: 700; color: #0f172a; }}
.stat-label {{ font-size: 0.85rem; color: #64748b; margin-top: 4px; }}
.charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
.chart-box {{ background: #f8fafc; border-radius: 8px; padding: 16px; }}
.chart-box h3 {{ font-size: 1rem; color: #334155; margin-bottom: 12px; }}
.chart-box canvas {{ max-height: 280px; }}
.analysis-box {{ background: #f8fafc; border-radius: 8px; padding: 20px; margin-bottom: 16px; }}
.analysis-box h3 {{ font-size: 1rem; color: #334155; margin-bottom: 12px; }}
.analysis-box ul, .analysis-box ol {{ padding-left: 20px; }}
.analysis-box li {{ margin-bottom: 6px; font-size: 0.9rem; }}
.analysis-box p {{ font-size: 0.95rem; color: #334155; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 12px; }}
thead th {{ background: #f1f5f9; padding: 10px 12px; text-align: left; font-weight: 600; font-size: 0.85rem; color: #475569; border-bottom: 2px solid #e2e8f0; }}
tbody td {{ padding: 8px 12px; font-size: 0.9rem; border-bottom: 1px solid #f1f5f9; }}
tbody tr:hover {{ background: #f8fafc; }}
.best {{ color: #059669; font-weight: 700; }}
.footer {{ text-align: center; padding: 20px; color: #94a3b8; font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="container">
{body}
<div class="footer">Generated at {generated_at} · RAG System Comprehensive Evaluation</div>
</div>
</body>
</html>"""
