"""Generate an HTML comparison report for no_reranker vs with_reranker."""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def load_metrics(suffix: str) -> dict:
    return json.loads((OUTPUT_DIR / f"metrics_annotation_results_{suffix}_optimize.json").read_text())

def load_annotations(suffix: str) -> dict:
    return json.loads((OUTPUT_DIR / f"annotation_results_{suffix}_optimize.json").read_text())

nr_metrics = load_metrics("no_reranker")
wr_metrics = load_metrics("with_reranker")
nr_ann = load_annotations("no_reranker")
wr_ann = load_annotations("with_reranker")

def fmt(v: float) -> str:
    return f"{v:.4f}"

def pct(v: float) -> str:
    return f"{v:.2%}"

K_VALS = [1, 3, 5, 7, 10]

# Build per-question table rows
def question_rows(ann: dict) -> str:
    rows = []
    for a in ann["annotations"]:
        qid = a["question_id"]
        q = a["question"]
        diff = a["difficulty"]
        rel = a["relevant_chunk_ids"]
        rel_count = len(rel)
        retrieved = [c["chunk_id"] for c in a["all_retrieved_chunks"]]

        def recall(k: int) -> float:
            if not rel:
                return 1.0
            top = set(retrieved[:k])
            hits = len(set(rel) & top)
            return hits / len(rel)

        def mrr():
            for rank, cid in enumerate(retrieved, 1):
                if cid in rel:
                    return 1.0 / rank
            return 0.0

        def hit(k: int) -> bool:
            if not rel:
                return True
            return bool(set(rel) & set(retrieved[:k]))

        b = {"easy": "badge-easy", "medium": "badge-medium", "hard": "badge-hard"}.get(diff, "badge-easy")
        r1 = pct(recall(1))
        r3 = pct(recall(3))
        r5 = pct(recall(5))
        r7 = pct(recall(7))
        r10 = pct(recall(10))
        mr = f"{mrr():.4f}"
        h5 = "Y" if hit(5) else "N"
        rows.append(f"<tr><td>{qid}</td><td class=\"q-text\" title=\"{q}\">{q[:50]}{'…' if len(q)>50 else ''}</td>"
                     f"<td><span class=\"badge {b}\">{diff}</span></td>"
                     f"<td>{rel_count}</td><td>{r1}</td><td>{r3}</td><td>{r5}</td><td>{r7}</td><td>{r10}</td>"
                     f"<td>{mr}</td><td>{h5}</td></tr>")
    return "\n".join(rows)

nr_rows = question_rows(nr_ann)
wr_rows = question_rows(wr_ann)

nr_ar = nr_metrics["avg_recall"]
wr_ar = wr_metrics["avg_recall"]
nr_hr = nr_metrics["hit_rate"]
wr_hr = wr_metrics["hit_rate"]

def ar_row(data: dict) -> str:
    return "".join(f"<td>{data[f'@{k}']:.4f}</td>" for k in K_VALS)

def hr_row(data: dict) -> str:
    return "".join(f"<td>{data[f'@{k}']:.4f}</td>" for k in K_VALS)

def bar_data(data: dict) -> str:
    return ", ".join(f"{data[f'@{k}']:.4f}" for k in K_VALS)

def bar_label(v: float) -> str:
    return f"{v:.2%}"

nr_total = nr_metrics["total_questions"]
wr_total = wr_metrics["total_questions"]

# Compute total relevant chunks
def total_relevant(ann_data: dict) -> int:
    return sum(len(a["relevant_chunk_ids"]) for a in ann_data["annotations"])

nr_relevant = total_relevant(nr_ann)
wr_relevant = total_relevant(wr_ann)

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG 检索召回对比评估</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f8fafc;color:#1e293b;line-height:1.6;padding:20px}}
.container{{max-width:1200px;margin:0 auto}}
.header{{text-align:center;padding:40px 0 20px}}
.header h1{{font-size:2rem;color:#0f172a}}
.header .subtitle{{color:#64748b;margin-top:8px}}
.section{{background:#fff;border-radius:12px;padding:32px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}}
.section h2{{font-size:1.4rem;color:#1e40af;margin-bottom:20px;padding-bottom:8px;border-bottom:2px solid #e2e8f0}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}}
.stat-card{{background:#f1f5f9;border-radius:8px;padding:20px;text-align:center}}
.stat-card.accent-blue{{border-left:4px solid #3b82f6}}
.stat-card.accent-green{{border-left:4px solid #10b981}}
.stat-card.accent-orange{{border-left:4px solid #f59e0b}}
.stat-card.accent-purple{{border-left:4px solid #8b5cf6}}
.stat-value{{font-size:1.8rem;font-weight:700}}
.stat-value.good{{color:#10b981}}
.stat-value.better{{color:#3b82f6}}
.stat-value.best{{color:#8b5cf6}}
.stat-label{{font-size:0.85rem;color:#64748b;margin-top:4px}}
.comparison-row{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px}}
.mode-box{{background:#f8fafc;border-radius:8px;padding:16px}}
.mode-box h3{{font-size:1rem;color:#334155;margin-bottom:12px;text-align:center;padding:4px 0}}
.mode-box h3.nr{{color:#3b82f6}}
.mode-box h3.wr{{color:#8b5cf6}}
.chart-box{{background:#f8fafc;border-radius:8px;padding:16px;margin-bottom:16px}}
.chart-box h3{{font-size:1rem;color:#334155;margin-bottom:12px}}
.chart-box canvas{{max-height:260px}}
table{{width:100%;border-collapse:collapse}}
thead th{{background:#f1f5f9;padding:10px 12px;text-align:left;font-weight:600;font-size:0.85rem;color:#475569;border-bottom:2px solid #e2e8f0}}
tbody td{{padding:8px 12px;font-size:0.9rem;border-bottom:1px solid #f1f5f9}}
tbody tr:hover{{background:#f8fafc}}
.q-text{{max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600}}
.badge-easy{{background:#d1fae5;color:#065f46}}
.badge-medium{{background:#fef3c7;color:#92400e}}
.badge-hard{{background:#fee2e2;color:#991b1b}}
.delta-up{{color:#10b981;font-weight:600}}
.delta-down{{color:#ef4444;font-weight:600}}
.footer{{text-align:center;padding:20px;color:#94a3b8;font-size:0.8rem}}
.table-scroll{{overflow-x:auto}}
.table-scroll table{{font-size:0.85rem}}
</style></head><body><div class="container">
<div class="header"><h1>RAG 系统评估报告 — 检索召回对比</h1><p class="subtitle">基于 LLM 语义判断标注（optimize版） · 启用 BAAI/bge-reranker-base</p></div>

<div class="section"><h2>核心指标对比</h2>
<div class="stats-grid">
<div class="stat-card accent-blue"><div class="stat-value good">{pct(nr_ar['@5'])}</div><div class="stat-label">无 Reranker · Recall@5</div></div>
<div class="stat-card accent-purple"><div class="stat-value best">{pct(wr_ar['@5'])}</div><div class="stat-label">有 Reranker · Recall@5</div></div>
<div class="stat-card accent-blue"><div class="stat-value good">{nr_metrics['avg_mrr']:.4f}</div><div class="stat-label">无 Reranker · MRR</div></div>
<div class="stat-card accent-purple"><div class="stat-value best">{wr_metrics['avg_mrr']:.4f}</div><div class="stat-label">有 Reranker · MRR</div></div>
<div class="stat-card accent-green"><div class="stat-value good">{pct(nr_hr['@5'])}</div><div class="stat-label">无 Reranker · Hit Rate@5</div></div>
<div class="stat-card accent-orange"><div class="stat-value best">{pct(wr_hr['@5'])}</div><div class="stat-label">有 Reranker · Hit Rate@5</div></div>
</div>

<div class="comparison-row">
<div class="mode-box"><h3 class="nr">📊 无 Reranker（{nr_relevant} 个相关 Chunk / {nr_total} 题）</h3>
<div class="metrics-table">
<table><thead><tr><th>指标</th>{"".join(f'<th>K={k}</th>' for k in K_VALS)}</tr></thead>
<tbody>
<tr><td><strong>Recall@K</strong></td>{ar_row(nr_ar)}</tr>
<tr><td><strong>Hit Rate@K</strong></td>{hr_row(nr_hr)}</tr>
</tbody></table></div>
<div class="chart-box"><h3>Recall@K</h3><canvas id="recallChartNR"></canvas></div>
<div class="chart-box"><h3>Hit Rate@K</h3><canvas id="hitRateChartNR"></canvas></div>
</div>
<div class="mode-box"><h3 class="wr">🚀 有 Reranker（{wr_relevant} 个相关 Chunk / {wr_total} 题）</h3>
<div class="metrics-table">
<table><thead><tr><th>指标</th>{"".join(f'<th>K={k}</th>' for k in K_VALS)}</tr></thead>
<tbody>
<tr><td><strong>Recall@K</strong></td>{ar_row(wr_ar)}</tr>
<tr><td><strong>Hit Rate@K</strong></td>{hr_row(wr_hr)}</tr>
</tbody></table></div>
<div class="chart-box"><h3>Recall@K</h3><canvas id="recallChartWR"></canvas></div>
<div class="chart-box"><h3>Hit Rate@K</h3><canvas id="hitRateChartWR"></canvas></div>
</div>
</div>

<div class="chart-box"><h3>Recall@K 对比</h3><canvas id="recallCompareChart" style="max-height:300px"></canvas></div>
</div>

<div class="section"><h2>按难度分组</h2>
<div class="comparison-row">
<div class="mode-box"><h3 class="nr">无 Reranker</h3>
<table><thead><tr><th>难度</th><th>数量</th><th>R@5</th><th>R@10</th><th>MRR</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{d}</td><td>{v["count"]}</td><td>{pct(v["avg_recall"]["@5"])}</td><td>{pct(v["avg_recall"]["@10"])}</td><td>{v["avg_mrr"]:.4f}</td></tr>' for d,v in sorted(nr_metrics['by_difficulty'].items(), key=lambda x: -x[1]['count']))}
</tbody></table></div>
<div class="mode-box"><h3 class="wr">有 Reranker</h3>
<table><thead><tr><th>难度</th><th>数量</th><th>R@5</th><th>R@10</th><th>MRR</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{d}</td><td>{v["count"]}</td><td>{pct(v["avg_recall"]["@5"])}</td><td>{pct(v["avg_recall"]["@10"])}</td><td>{v["avg_mrr"]:.4f}</td></tr>' for d,v in sorted(wr_metrics['by_difficulty'].items(), key=lambda x: -x[1]['count']))}
</tbody></table></div>
</div>
</div>

<div class="section"><h2>逐题对比 · 无 Reranker</h2>
<div class="table-scroll"><table><thead><tr><th>ID</th><th>问题</th><th>难度</th><th>相关数</th><th>R@1</th><th>R@3</th><th>R@5</th><th>R@7</th><th>R@10</th><th>MRR</th><th>Hit@5</th></tr></thead><tbody>
{nr_rows}
</tbody></table></div></div>

<div class="section"><h2>逐题对比 · 有 Reranker</h2>
<div class="table-scroll"><table><thead><tr><th>ID</th><th>问题</th><th>难度</th><th>相关数</th><th>R@1</th><th>R@3</th><th>R@5</th><th>R@7</th><th>R@10</th><th>MRR</th><th>Hit@5</th></tr></thead><tbody>
{wr_rows}
</tbody></table></div></div>

<div class="footer">基于 LLM 语义判断标注 · 生成于 2026-05-18</div></div>
<script>
new Chart(document.getElementById('recallChartNR'),{{type:'line',data:{{labels:["K=1","K=3","K=5","K=7","K=10"],datasets:[{{label:'Recall@K',data:[{bar_data(nr_ar)}],borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,0.1)',fill:true,tension:0.3}}]}},options:{{responsive:true,scales:{{y:{{min:0,max:1}}}}}}}});
new Chart(document.getElementById('hitRateChartNR'),{{type:'bar',data:{{labels:["K=1","K=3","K=5","K=7","K=10"],datasets:[{{label:'Hit Rate',data:[{bar_data(nr_hr)}],backgroundColor:'#3b82f6'}}]}},options:{{responsive:true,scales:{{y:{{min:0,max:1}}}}}}}});
new Chart(document.getElementById('recallChartWR'),{{type:'line',data:{{labels:["K=1","K=3","K=5","K=7","K=10"],datasets:[{{label:'Recall@K',data:[{bar_data(wr_ar)}],borderColor:'#8b5cf6',backgroundColor:'rgba(139,92,246,0.1)',fill:true,tension:0.3}}]}},options:{{responsive:true,scales:{{y:{{min:0,max:1}}}}}}}});
new Chart(document.getElementById('hitRateChartWR'),{{type:'bar',data:{{labels:["K=1","K=3","K=5","K=7","K=10"],datasets:[{{label:'Hit Rate',data:[{bar_data(wr_hr)}],backgroundColor:'#8b5cf6'}}]}},options:{{responsive:true,scales:{{y:{{min:0,max:1}}}}}}}});
new Chart(document.getElementById('recallCompareChart'),{{type:'line',data:{{labels:["K=1","K=3","K=5","K=7","K=10"],datasets:[
{{label:'无 Reranker',data:[{bar_data(nr_ar)}],borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,0.05)',fill:false,tension:0.3,borderWidth:2,borderDash:[5,5]}},
{{label:'有 Reranker',data:[{bar_data(wr_ar)}],borderColor:'#8b5cf6',backgroundColor:'rgba(139,92,246,0.05)',fill:false,tension:0.3,borderWidth:3}}
]}},options:{{responsive:true,scales:{{y:{{min:0,max:1}}}}}}}});
</script></body></html>
"""

path = OUTPUT_DIR / "retrieval_evaluation_optimize_comparison.html"
path.write_text(html, encoding="utf-8")
print(f"Report saved to {path}")
