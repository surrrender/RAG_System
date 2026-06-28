"""
QA 问答效果评估脚本
====================
对 40 个问题逐一通过 RAG 管道生成答案，计算 BLEU/ROUGE 及 LLM-as-Judge 评分，
生成 HTML 评估报告。

用法:
    source .venv/bin/activate
    python -m evaluation.run_qa_evaluation
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm
from evaluation.evaluators.metrics import compute_bleu, compute_rouge

# ---------------------------------------------------------------------------
# 1. Parse reference answers from markdown
# ---------------------------------------------------------------------------
REF_PATH = REPO_ROOT / "evaluation" / "data" / "测试问题和答案.md"

def parse_reference_answers(md_path: Path) -> dict[str, str]:
    text = md_path.read_text(encoding="utf-8")
    q_blocks = re.split(r'(?=## \d+)', text)
    refs: dict[str, str] = {}
    q_index = 0
    for block in q_blocks:
        if not block.strip():
            continue
        # Extract answer (everything after the first question line, before next ##)
        lines = block.strip().split('\n')
        # Find the question number like "## 1."
        q_match = re.match(r'##\s*(\d+)\.', lines[0])
        if not q_match:
            continue
        q_num = int(q_match.group(1))
        q_id = f"q{q_num:03d}"
        # Answer content = everything after the first blank line following the question
        answer_lines = []
        in_answer = False
        for line in lines[1:]:
            if line.strip() == '' and not in_answer:
                in_answer = True
                continue
            if in_answer:
                if line.startswith('#'):
                    break
                answer_lines.append(line)
        answer = '\n'.join(answer_lines).strip()
        refs[q_id] = answer
    return refs

# ---------------------------------------------------------------------------
# 2. Load 40 questions from the annotation data
# ---------------------------------------------------------------------------
def load_questions() -> list[dict]:
    with open(REPO_ROOT / "outputs" / "annotation_results_with_reranker_llm.json", "r") as f:
        data = json.load(f)
    return data["annotations"]

# ---------------------------------------------------------------------------
# 3. LLM prompts
# ---------------------------------------------------------------------------
FAITHFULNESS_PROMPT = """评估以下回答是否忠实于给定的检索上下文。请根据上下文判断回答中的每个声明是否都有据可查。

检索上下文（从知识库中检索出的相关文档片段）：
{context}

回答：
{answer}

评分标准（1-5分）：
1分：回答内容与上下文完全无关，纯粹是臆测
2分：回答仅有少量内容基于上下文，大部分无依据
3分：回答部分基于上下文，但存在明显的无依据扩展
4分：回答基本基于上下文，仅有极少量细节超出上下文范围
5分：回答完全基于上下文，所有声明都能在上下文中找到明确依据

请只输出 JSON：{{"score": 1-5, "reason": "简短理由"}}"""

HUMAN_SCORE_PROMPT = """请从以下三个维度对回答进行评分（每项1-5分）：

问题：{question}

参考答案：{reference}

实际回答：{answer}

评分维度：
1. 相关性（Relevance）：回答是否切中问题要点，没有跑题
2. 完整性（Completeness）：回答是否涵盖了问题的关键方面
3. 准确性（Accuracy）：回答中的技术细节是否正确

请只输出 JSON：
{{"relevance": 1-5, "completeness": 1-5, "accuracy": 1-5, "overall": 1-5, "reason": "简短理由"}}"""

def _call_llm(generator, prompt: str) -> str:
    return generator.generate(prompt)

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None

# ---------------------------------------------------------------------------
# 4. Build HTML report
# ---------------------------------------------------------------------------
def generate_html_report(results: list[dict], avg_metrics: dict, output_path: Path) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # Prepare chart data
    bleu_labels = ['BLEU-1', 'BLEU-2', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    bleu_values = [
        avg_metrics['avg_bleu_1'],
        avg_metrics['avg_bleu_2'],
        avg_metrics['avg_bleu_4'],
        avg_metrics['avg_rouge_1_f'],
        avg_metrics['avg_rouge_2_f'],
        avg_metrics['avg_rouge_l_f'],
    ]

    # Faithfulness distribution
    faith_dist = [0, 0, 0, 0, 0]
    for r in results:
        s = max(1, min(5, r['faithfulness_score']))
        faith_dist[s - 1] += 1

    # Human score distribution
    human_labels = ['Faithfulness', 'Relevance', 'Completeness', 'Accuracy', 'Overall']
    human_values = [
        avg_metrics['avg_faithfulness'],
        avg_metrics['avg_relevance'],
        avg_metrics['avg_completeness'],
        avg_metrics['avg_accuracy'],
        avg_metrics['avg_overall'],
    ]

    # BLEU by difficulty
    bleu_by_diff = {}
    for d in ['easy', 'medium', 'hard']:
        grp = [r for r in results if r['difficulty'] == d]
        n = max(len(grp), 1)
        bleu_by_diff[d] = {
            'bleu_1': sum(r['bleu_1'] for r in grp) / n,
            'bleu_2': sum(r['bleu_2'] for r in grp) / n,
            'bleu_4': sum(r['bleu_4'] for r in grp) / n,
        }

    diff_keys = [k for k in ['easy', 'medium', 'hard'] if k in bleu_by_diff]

    # Count per-row
    total_q = len(results)

    rows_html = ""
    for r in results:
        diff_badge = f'<span class="badge badge-{r["difficulty"]}">{r["difficulty"]}</span>'
        q_short = r['question'][:55] + '...' if len(r['question']) > 55 else r['question']
        overall_display = f'{r["overall_score"]:.1f}' if r["overall_score"] else '-'
        rows_html += f'''        <tr>
            <td>{r['qid']}</td>
            <td class="q-text" title="{r['question']}">{q_short}</td>
            <td>{diff_badge}</td>
            <td>{r['bleu_1']:.3f}</td>
            <td>{r['bleu_4']:.3f}</td>
            <td>{r['rouge_l_f']:.3f}</td>
            <td>{r['faithfulness_score']}</td>
            <td>{overall_display}</td>
        </tr>
        
'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 系统评估报告 — 微信小程序开发文档 (问答效果评估)</title>
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

    <div class="header">
        <h1>RAG 系统评估报告 — 问答效果评估</h1>
        <p class="subtitle">微信小程序开发文档 · qwen3:8b (RAG) · {now}</p>
    </div>

    <div class="section">
        <h2>1. 数据集统计</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_q}</div>
                <div class="stat-label">测试 QA 对数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([r for r in results if r['generated_answer'] != '（生成失败）'])}/{total_q}</div>
                <div class="stat-label">成功生成数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum(r['retrieval_count'] for r in results) / max(total_q, 1):.0f}</div>
                <div class="stat-label">平均检索 Chunk 数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_metrics['avg_overall']:.1f}/5</div>
                <div class="stat-label">综合评分</div>
            </div>
        </div>
        <div class="charts-row">
            <div class="chart-box">
                <h3>问题难度分布</h3>
                <canvas id="difficultyChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>Faithfulness 分数分布</h3>
                <canvas id="faithfulnessDistChart"></canvas>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>2. 问答效果评估</h2>

        <div class="stats-grid">
            <div class="stat-card accent-blue">
                <div class="stat-value">{avg_metrics['avg_bleu_1']:.1%}</div>
                <div class="stat-label">Avg BLEU-1</div>
            </div>
            <div class="stat-card accent-green">
                <div class="stat-value">{avg_metrics['avg_rouge_l_f']:.1%}</div>
                <div class="stat-label">Avg ROUGE-L</div>
            </div>
            <div class="stat-card accent-orange">
                <div class="stat-value">{avg_metrics['avg_faithfulness']:.1f}/5</div>
                <div class="stat-label">Avg Faithfulness</div>
            </div>
            <div class="stat-card accent-purple">
                <div class="stat-value">{avg_metrics['avg_overall']:.1f}/5</div>
                <div class="stat-label">Avg Human Score</div>
            </div>
        </div>

        <div class="metrics-table">
            <h3>核心指标汇总</h3>
            <table>
                <thead><tr><th>指标</th><th>值</th></tr></thead>
                <tbody>
                    <tr><td>BLEU-1</td><td>{avg_metrics['avg_bleu_1']:.4f}</td></tr>
                    <tr><td>BLEU-2</td><td>{avg_metrics['avg_bleu_2']:.4f}</td></tr>
                    <tr><td>BLEU-4</td><td>{avg_metrics['avg_bleu_4']:.4f}</td></tr>
                    <tr><td>ROUGE-1 F</td><td>{avg_metrics['avg_rouge_1_f']:.4f}</td></tr>
                    <tr><td>ROUGE-2 F</td><td>{avg_metrics['avg_rouge_2_f']:.4f}</td></tr>
                    <tr><td>ROUGE-L F</td><td>{avg_metrics['avg_rouge_l_f']:.4f}</td></tr>
                    <tr><td>Faithfulness</td><td>{avg_metrics['avg_faithfulness']:.2f} / 5</td></tr>
                    <tr><td>Relevance</td><td>{avg_metrics['avg_relevance']:.2f} / 5</td></tr>
                    <tr><td>Completeness</td><td>{avg_metrics['avg_completeness']:.2f} / 5</td></tr>
                    <tr><td>Accuracy</td><td>{avg_metrics['avg_accuracy']:.2f} / 5</td></tr>
                    <tr><td>Overall Human Score</td><td>{avg_metrics['avg_overall']:.2f} / 5</td></tr>
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
                <canvas id="faithfulnessDistChart2"></canvas>
            </div>
            <div class="chart-box">
                <h3>不同难度 BLEU-4 对比</h3>
                <canvas id="bleuDifficultyChart"></canvas>
            </div>
        </div>

        <h3>逐题问答结果</h3>
        <div class="table-scroll">
            <table class="detail-table">
                <thead><tr><th>ID</th><th>问题</th><th>难度</th><th>BLEU-1</th><th>BLEU-4</th><th>ROUGE-L</th><th>Faith.</th><th>Overall</th></tr></thead>
                <tbody>
{rows_html}        </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>3. 综合分析</h2>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len([r for r in results if r.get('overall_score', 0) >= 4])}/{total_q}</div>
                <div class="stat-label">评分 ≥4 问题数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([r for r in results if r.get('faithfulness_score', 0) >= 4])}/{total_q}</div>
                <div class="stat-label">Faithfulness ≥4 问题数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_metrics['avg_overall']:.1f}</div>
                <div class="stat-label">系统整体评分</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_metrics['avg_bleu_4']:.1%}</div>
                <div class="stat-label">Avg BLEU-4</div>
            </div>
        </div>

        <div class="analysis-box">
            <h3>关键发现</h3>
            <ul>
                <li>BLEU-1 平均 {avg_metrics['avg_bleu_1']:.1%}，BLEU-4 平均 {avg_metrics['avg_bleu_4']:.1%}，表明回答与参考答案的词汇重叠度</li>
                <li>ROUGE-L 平均 {avg_metrics['avg_rouge_l_f']:.1%}，反映生成回答与参考答案的文本相似度</li>
                <li>Faithfulness 评分 {avg_metrics['avg_faithfulness']:.1f}/5，表明生成回答基于检索内容的忠实度</li>
                <li>综合评分 {avg_metrics['avg_overall']:.1f}/5，反映回答的整体质量</li>
            </ul>
        </div>
    </div>

<div class="footer">Generated at {now} · RAG System Evaluation - QA</div>
</div>
<script>
'''

    diff_labels = json.dumps(diff_keys)
    diff_counts = json.dumps([len([r for r in results if r['difficulty'] == d]) for d in diff_keys])
    faith_dist_json = json.dumps(faith_dist)

    html += f'''    new Chart(document.getElementById('difficultyChart'), {{
        type: 'doughnut',
        data: {{ labels: {diff_labels}, datasets: [{{ data: {diff_counts}, backgroundColor: ['#10b981','#f59e0b','#ef4444'] }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});
    new Chart(document.getElementById('faithfulnessDistChart'), {{
        type: 'bar',
        data: {{
            labels: ['1分','2分','3分','4分','5分'],
            datasets: [{{ label: '问题数', data: {faith_dist_json}, backgroundColor: ['#ef4444','#f97316','#f59e0b','#10b981','#3b82f6'] }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
    }});
    new Chart(document.getElementById('bleuRougeChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(bleu_labels)},
            datasets: [{{ label: 'Score', data: {json.dumps(bleu_values)}, backgroundColor: ['#3b82f6','#60a5fa','#93c5fd','#10b981','#34d399','#6ee7b7'] }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0 }} }} }}
    }});
    new Chart(document.getElementById('humanChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(human_labels)},
            datasets: [{{ label: 'Avg Score', data: {json.dumps(human_values)}, backgroundColor: ['#8b5cf6','#3b82f6','#10b981','#f59e0b','#ef4444'] }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0, max: 5 }} }} }}
    }});
    new Chart(document.getElementById('faithfulnessDistChart2'), {{
        type: 'bar',
        data: {{
            labels: ['1分','2分','3分','4分','5分'],
            datasets: [{{ label: '问题数', data: {faith_dist_json}, backgroundColor: ['#ef4444','#f97316','#f59e0b','#10b981','#3b82f6'] }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
    }});
    _chartBleuDifficulty({json.dumps(bleu_by_diff)});
</script>
<script>
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
        new Chart(canvas, {{ type: 'bar', data: {{ labels: keys, datasets: datasets }}, options: {{ responsive: true, scales: {{ y: {{ min: 0 }} }} }} }});
    }}
}}
</script>
</body>
</html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\n报告已生成: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# 5. Main evaluation loop
# ---------------------------------------------------------------------------
def main() -> None:
    refs = parse_reference_answers(REF_PATH)
    questions = load_questions()
    print(f"已加载 {len(refs)} 个参考答案，{len(questions)} 个问题")

    # Build service (RAG pipeline + LLM generation)
    from llm.config import load_settings
    from llm.service import build_service
    from llm.generator import OllamaGenerator

    settings = load_settings()
    service = build_service(settings)
    generator = OllamaGenerator(
        host=settings.ollama_host or "http://127.0.0.1:11434",
        model=settings.generation_model or "qwen3:8b",
        timeout=120.0,
    )

    # Check for existing results to resume
    results_path = REPO_ROOT / "outputs" / "qa_eval_results_optimize.json"
    existing_results: list[dict] = []
    if results_path.exists():
        try:
            existing_results = json.loads(results_path.read_text(encoding="utf-8"))
            print(f"找到已有结果: {len(existing_results)} 题已处理")
        except Exception:
            existing_results = []

    processed_ids = {r['qid'] for r in existing_results}
    results = list(existing_results)

    for idx, q in enumerate(tqdm(questions, desc="QA Evaluation")):
        qid = q['question_id']
        question = q['question']
        category = q['category']
        difficulty = q['difficulty']

        if qid in processed_ids:
            continue

        reference = refs.get(qid, "")
        generated_answer = "（生成失败）"
        retrieval_count = 0
        context = "（无检索结果）"

        try:
            # top_k=10 with reranker enabled (default)
            answer_result = service.answer_question(question, top_k=5)
            generated_answer = answer_result.answer
            retrieval_count = answer_result.retrieval_count
            context_parts = []
            for citation in answer_result.citations:
                text = citation.text or ""
                if text:
                    context_parts.append(f"[{citation.chunk_id}] {text[:800]}")
            context = "\n\n".join(context_parts) if context_parts else "（无检索结果）"
        except Exception as exc:
            print(f"\n  [{idx+1}/{len(questions)}] {qid} 生成失败: {exc}")

        bleu = compute_bleu([reference], generated_answer)
        rouge = compute_rouge([reference], generated_answer)

        faithfulness = 3
        relevance = 3
        completeness = 3
        accuracy = 3
        overall = 3.0

        if generated_answer != "（生成失败）":
            try:
                f_prompt = FAITHFULNESS_PROMPT.format(context=context, answer=generated_answer)
                f_response = _call_llm(generator, f_prompt)
                f_data = _extract_json(f_response)
                if f_data:
                    faithfulness = int(f_data.get("score", 3))
                time.sleep(0.5)
            except Exception:
                pass

            try:
                h_prompt = HUMAN_SCORE_PROMPT.format(
                    question=question,
                    reference=reference,
                    answer=generated_answer,
                )
                h_response = _call_llm(generator, h_prompt)
                h_data = _extract_json(h_response)
                if h_data:
                    relevance = int(h_data.get("relevance", 3))
                    completeness = int(h_data.get("completeness", 3))
                    accuracy = int(h_data.get("accuracy", 3))
                    overall = float(h_data.get("overall", 3.0))
                time.sleep(0.5)
            except Exception:
                pass

        result = {
            "qid": qid,
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "reference_answer": reference,
            "generated_answer": generated_answer,
            "bleu_1": bleu["bleu_1"],
            "bleu_2": bleu["bleu_2"],
            "bleu_4": bleu["bleu_4"],
            "rouge_1_f": rouge["rouge_1_f"],
            "rouge_2_f": rouge["rouge_2_f"],
            "rouge_l_f": rouge["rouge_l_f"],
            "faithfulness_score": faithfulness,
            "relevance_score": relevance,
            "completeness_score": completeness,
            "accuracy_score": accuracy,
            "overall_score": overall,
            "retrieval_count": retrieval_count,
        }
        results.append(result)
        processed_ids.add(qid)

        # Save intermediate results
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        tqdm.write(f"  [{idx+1}/{len(questions)}] {qid}: BLEU-1={bleu['bleu_1']:.3f}, BLEU-4={bleu['bleu_4']:.3f}, ROUGE-L={rouge['rouge_l_f']:.3f}, Faith={faithfulness}, Overall={overall}")

    # Compute averages
    n = max(len(results), 1)
    avg_metrics = {
        "avg_bleu_1": sum(r['bleu_1'] for r in results) / n,
        "avg_bleu_2": sum(r['bleu_2'] for r in results) / n,
        "avg_bleu_4": sum(r['bleu_4'] for r in results) / n,
        "avg_rouge_1_f": sum(r['rouge_1_f'] for r in results) / n,
        "avg_rouge_2_f": sum(r['rouge_2_f'] for r in results) / n,
        "avg_rouge_l_f": sum(r['rouge_l_f'] for r in results) / n,
        "avg_faithfulness": sum(r['faithfulness_score'] for r in results) / n,
        "avg_relevance": sum(r['relevance_score'] for r in results) / n,
        "avg_completeness": sum(r['completeness_score'] for r in results) / n,
        "avg_accuracy": sum(r['accuracy_score'] for r in results) / n,
        "avg_overall": sum(r['overall_score'] for r in results) / n,
    }

    print("\n" + "=" * 60)
    print("问答效果评估汇总")
    print("=" * 60)
    for k, v in avg_metrics.items():
        display = f"{v:.2%}" if "bleu" in k or "rouge" in k else f"{v:.2f}"
        print(f"  {k}: {display}")
    print(f"  总题数: {n}")

    # Generate HTML report
    output_html = REPO_ROOT / "evaluation" / "outputs" / "qa_evaluation_report_optimize.html"
    generate_html_report(results, avg_metrics, output_html)

    # Also save a merged comprehensive report combining retrieval + QA
    print(f"\n完整结果已保存到: {results_path.resolve()}")


if __name__ == "__main__":
    main()
