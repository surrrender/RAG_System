"""
Latency benchmark script.
Randomly selects 5 questions, calls the RAG service with top_k=10,
measures latency metrics, and generates an HTML dashboard.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_questions() -> list[dict]:
    with open(REPO_ROOT / "outputs" / "annotation_results_with_reranker_llm.json") as f:
        data = json.load(f)
    return data["annotations"]


def build_service():
    from llm.config import load_settings
    from llm.service import build_service

    settings = load_settings()
    return build_service(settings)


def run_benchmark(questions: list[dict], top_k: int = 5) -> list[dict]:
    service = build_service()
    samples = []

    for idx, q in enumerate(questions):
        qid = q["question_id"]
        question = q["question"]
        print(f"  [{idx + 1}/{len(questions)}] {qid}: {question[:50]}...", flush=True)

        start = time.perf_counter()
        first_token_at: float | None = None
        server_retrieval_ms: float | None = None
        server_embed_ms: float | None = None
        server_vector_search_ms: float | None = None
        server_rerank_ms: float | None = None
        server_prompt_build_ms: float | None = None
        done_at: float | None = None

        for event in service.stream_answer_question(question, top_k=top_k):
            if event["event"] == "meta":
                data = event["data"]
                retrieval_finished_at = data.get("retrieval_finished_at_ms", 0)
                server_started_at = data.get("server_started_at_ms", 0)
                server_embed_ms = data.get("server_embed_ms")
                server_vector_search_ms = data.get("server_vector_search_ms")
                server_rerank_ms = data.get("server_rerank_ms")
                server_prompt_build_ms = data.get("server_prompt_build_ms")
                if retrieval_finished_at and server_started_at is not None:
                    server_retrieval_ms = retrieval_finished_at - server_started_at
            elif event["event"] == "delta":
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            elif event["event"] == "done" or event["event"] == "error":
                done_at = time.perf_counter()
                break

        if done_at is None:
            done_at = time.perf_counter()

        time_to_first = (first_token_at - start) * 1000 if first_token_at else None
        time_to_full = (done_at - start) * 1000

        sample = {
            "qid": qid,
            "question": question,
            "time_to_first_visible_char_ms": time_to_first,
            "time_to_full_visible_answer_ms": time_to_full,
            "server_retrieval_ms": server_retrieval_ms,
            "server_embed_ms": server_embed_ms,
            "server_vector_search_ms": server_vector_search_ms,
            "server_rerank_ms": server_rerank_ms,
            "server_prompt_build_ms": server_prompt_build_ms,
        }
        samples.append(sample)

        print(
            f"    first_char={time_to_first:.1f}ms  "
            f"full={time_to_full:.1f}ms  "
            f"server_retrieval={server_retrieval_ms:.1f}ms"
        )

    return samples


def generate_html(samples: list[dict], output_path: Path) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    avg = {
        "first": sum(s["time_to_first_visible_char_ms"] for s in samples) / len(samples),
        "full": sum(s["time_to_full_visible_answer_ms"] for s in samples) / len(samples),
        "server": sum(
            s["server_retrieval_ms"] for s in samples if s["server_retrieval_ms"] is not None
        )
        / len(samples),
        "ratio": sum(
            s["time_to_first_visible_char_ms"] / s["time_to_full_visible_answer_ms"]
            for s in samples
        )
        / len(samples),
    }

    raw_data_items = []
    for s in samples:
        ratio = s["time_to_first_visible_char_ms"] / s["time_to_full_visible_answer_ms"]
        item = {
            "time_to_first_visible_char_ms": round(s["time_to_first_visible_char_ms"], 1),
            "time_to_full_visible_answer_ms": round(s["time_to_full_visible_answer_ms"], 1),
            "server_retrieval_ms": (
                round(s["server_retrieval_ms"], 2) if s["server_retrieval_ms"] is not None else 0
            ),
            "server_embed_ms": (
                round(s["server_embed_ms"], 3) if s["server_embed_ms"] is not None else 0
            ),
            "server_vector_search_ms": (
                round(s["server_vector_search_ms"], 3)
                if s["server_vector_search_ms"] is not None
                else 0
            ),
            "server_rerank_ms": (
                round(s["server_rerank_ms"], 3) if s["server_rerank_ms"] is not None else 0
            ),
            "server_prompt_build_ms": (
                round(s["server_prompt_build_ms"], 3)
                if s["server_prompt_build_ms"] is not None
                else 0
            ),
        }
        raw_data_items.append(json.dumps(item, ensure_ascii=False))

    raw_data_js = ",\n      ".join(raw_data_items)
    qlist_json = json.dumps(
        [{"qid": s["qid"], "question": s["question"]} for s in samples],
        ensure_ascii=False,
        indent=2,
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG Latency Benchmark (top_k=10)</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: rgba(255, 251, 244, 0.88);
      --panel-border: rgba(96, 71, 52, 0.12);
      --text: #2f241c;
      --muted: #7a6758;
      --grid: rgba(88, 63, 43, 0.12);
      --first-char: #d95d39;
      --full-answer: #2a7f62;
      --server: #457b9d;
      --ratio: #c08429;
      --embed: #d95d39;
      --vector-search: #457b9d;
      --rerank: #2a7f62;
      --prompt-build: #c08429;
      --shadow: 0 18px 48px rgba(74, 54, 40, 0.1);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Avenir Next","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; color: var(--text); background: radial-gradient(circle at top left, rgba(217,93,57,0.16), transparent 32%), radial-gradient(circle at top right, rgba(69,123,157,0.15), transparent 28%), linear-gradient(180deg,#f9f5ee 0%,var(--bg) 100%); }}
    .page {{ width: min(1360px, calc(100% - 32px)); margin: 32px auto 48px; }}
    .hero {{ padding: 28px; border: 1px solid var(--panel-border); border-radius: calc(var(--radius) + 6px); background: linear-gradient(135deg, rgba(255,249,240,0.92), rgba(255,255,255,0.82)); box-shadow: var(--shadow); backdrop-filter: blur(14px); }}
    h1,h2 {{ margin: 0; font-weight: 700; letter-spacing: 0.01em; }}
    h1 {{ font-size: clamp(28px,4vw,44px); }}
    h2 {{ font-size: 22px; margin-bottom: 10px; }}
    .subtitle {{ margin-top: 12px; color: var(--muted); font-size: 15px; line-height: 1.7; max-width: 920px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-top: 24px; }}
    .stat-card,.panel {{ border: 1px solid var(--panel-border); border-radius: var(--radius); background: var(--panel); box-shadow: var(--shadow); backdrop-filter: blur(10px); }}
    .stat-card {{ padding: 18px 20px; }}
    .stat-label {{ color: var(--muted); font-size: 13px; margin-bottom: 10px; }}
    .stat-value {{ font-size: clamp(24px,2.2vw,34px); font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 20px; }}
    .panel {{ padding: 22px; }}
    .panel-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 16px; }}
    .panel-desc {{ color: var(--muted); font-size: 14px; line-height: 1.7; max-width: 820px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 13px; margin-top: 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
    .chart-wrap {{ width: 100%; overflow-x: auto; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .table-wrap {{ overflow-x: auto; margin-top: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th,td {{ padding: 12px 10px; border-bottom: 1px solid rgba(96,71,52,0.1); text-align: right; white-space: nowrap; }}
    th:first-child,td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .footer-note {{ margin-top: 18px; color: var(--muted); font-size: 13px; }}
    .question-cell {{ max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    @media (max-width: 900px) {{ .stats {{ grid-template-columns: repeat(2,minmax(0,1fr)); }} }}
    @media (max-width: 640px) {{ .page {{ width: min(100%-20px,100%); margin: 18px auto 28px; }} .hero,.panel {{ padding: 18px; }} .stats {{ grid-template-columns: 1fr; }} .panel-head {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>RAG Latency Benchmark (top_k=10)</h1>
      <p class="subtitle">
        随机选取 5 个测试问题，通过 Python 后端直接调用 <code>stream_answer_question</code>，
        测量 <code>time_to_first_visible_char_ms</code>、
        <code>time_to_full_visible_answer_ms</code>、
        <code>server_retrieval_ms</code> 三项耗时，并计算检索耗时占比。
      </p>
      <p class="subtitle">生成时间：{now} · top_k=10 · 随机样本</p>
      <div class="stats" id="summaryCards"></div>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>5 组耗时对比柱状图</h2>
            <div class="panel-desc">每组数据用三根柱子并排展示，方便直接比较首字可见时间、完整答案时间和服务端检索时间。</div>
          </div>
        </div>
        <div class="legend">
          <span><i class="swatch" style="background: var(--first-char);"></i> 首字可见时间</span>
          <span><i class="swatch" style="background: var(--full-answer);"></i> 完整答案时间</span>
          <span><i class="swatch" style="background: var(--server);"></i> 服务端检索时间</span>
        </div>
        <div class="chart-wrap">
          <svg id="groupedChart" viewBox="0 0 1220 520" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>检索耗时占比</h2>
            <div class="panel-desc">这里展示每组 <code>server_retrieval_ms</code> 占 <code>time_to_full_visible_answer_ms</code> 的比例，单位为百分比。</div>
          </div>
        </div>
        <div class="chart-wrap">
          <svg id="ratioChart" viewBox="0 0 1220 420" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>5 组平均值柱状图</h2>
            <div class="panel-desc">最后汇总四个指标的平均值。前三个是毫秒，最后一个是占比平均值，因此右侧单独使用百分比刻度。</div>
          </div>
        </div>
        <div class="chart-wrap">
          <svg id="averageChart" viewBox="0 0 1040 460" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>明细数据表</h2>
            <div class="panel-desc">每题的完整指标数据。</div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>题目</th>
                <th>首字可见 (ms)</th>
                <th>完整答案 (ms)</th>
                <th>服务端检索 (ms)</th>
                <th>检索耗时占比</th>
              </tr>
            </thead>
            <tbody id="detailRows"></tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>平均阶段耗时占比饼图</h2>
            <div class="panel-desc">对所有样本分别计算 <code>embed</code>、<code>vector_search</code>、<code>reranker</code>、<code>prompt_build</code> 的平均耗时，再汇总成一个饼图。</div>
          </div>
        </div>
        <div class="legend" id="stageBreakdownLegend"></div>
        <div class="chart-wrap">
          <svg id="stageBreakdownPieChart" viewBox="0 0 1220 520" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
        <div class="footer-note" id="stageBreakdownSummary"></div>
      </section>
    </div>
  </div>

  <script>
    const questions = {qlist_json};
    const rawData = [
      {raw_data_js}
    ].map((item, index) => ({{
      ...item,
      group: questions[index].qid,
      question: questions[index].question,
      ratio: item.server_retrieval_ms / item.time_to_full_visible_answer_ms
    }}));

    const colors = {{
      first: "#d95d39", full: "#2a7f62", server: "#457b9d", ratio: "#c08429",
      embed: "#d95d39", vectorSearch: "#457b9d", rerank: "#2a7f62", promptBuild: "#c08429",
      grid: "rgba(88, 63, 43, 0.12)", axis: "#7a6758", text: "#2f241c"
    }};

    const stageMetricSeries = [
      {{ key: "server_embed_ms", label: "Embed", color: colors.embed }},
      {{ key: "server_vector_search_ms", label: "Vector Search", color: colors.vectorSearch }},
      {{ key: "server_rerank_ms", label: "Reranker", color: colors.rerank }},
      {{ key: "server_prompt_build_ms", label: "Prompt Build", color: colors.promptBuild }}
    ];

    const average = rawData.reduce((acc, item) => {{
      acc.first += item.time_to_first_visible_char_ms;
      acc.full += item.time_to_full_visible_answer_ms;
      acc.server += item.server_retrieval_ms;
      acc.ratio += item.ratio;
      return acc;
    }}, {{ first: 0, full: 0, server: 0, ratio: 0 }});

    average.first /= rawData.length;
    average.full /= rawData.length;
    average.server /= rawData.length;
    average.ratio /= rawData.length;

    const stageBreakdownData = stageMetricSeries
      .map((series) => {{
        const values = rawData.map((item) => item[series.key]).filter((v) => typeof v === "number" && Number.isFinite(v) && v > 0);
        if (!values.length) return null;
        return {{ ...series, averageMs: values.reduce((s, v) => s + v, 0) / values.length }};
      }})
      .filter(Boolean);

    const stageBreakdownTotalMs = stageBreakdownData.reduce((sum, item) => sum + item.averageMs, 0);

    function fmtMs(v) {{ return v.toFixed(1) + " ms"; }}
    function fmtPct(v) {{ return (v * 100).toFixed(2) + "%"; }}
    function polarToCartesian(cx, cy, r, deg) {{
      const rad = ((deg - 90) * Math.PI) / 180;
      return {{ x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }};
    }}
    function describeArc(cx, cy, r, sa, ea) {{
      const s = polarToCartesian(cx, cy, r, ea);
      const e = polarToCartesian(cx, cy, r, sa);
      const laf = ea - sa <= 180 ? "0" : "1";
      return ["M",cx,cy,"L",s.x,s.y,"A",r,r,0,laf,0,e.x,e.y,"Z"].join(" ");
    }}
    function makeSvg(tag, attrs) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs || {{}}).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    }}
    function addText(svg, x, y, text, attrs) {{
      const t = makeSvg("text", {{ x, y, ...(attrs || {{}}) }});
      t.textContent = text;
      svg.appendChild(t);
      return t;
    }}
    function drawAxes(svg, w, h, m, maxV, tickC, fmt) {{
      const pw = w - m.left - m.right, ph = h - m.top - m.bottom;
      for (let i = 0; i <= tickC; i++) {{
        const val = maxV * (i / tickC);
        const y = m.top + ph - (val / maxV) * ph;
        svg.appendChild(makeSvg("line", {{ x1: m.left, y1: y, x2: m.left + pw, y2: y, stroke: colors.grid, "stroke-width": 1 }}));
        addText(svg, m.left - 10, y + 4, fmt(val), {{ "text-anchor": "end", fill: colors.axis, "font-size": 12 }});
      }}
      svg.appendChild(makeSvg("line", {{ x1: m.left, y1: m.top, x2: m.left, y2: m.top + ph, stroke: colors.axis, "stroke-width": 1.2 }}));
      svg.appendChild(makeSvg("line", {{ x1: m.left, y1: m.top + ph, x2: m.left + pw, y2: m.top + ph, stroke: colors.axis, "stroke-width": 1.2 }}));
      return {{ pw, ph }};
    }}

    function drawGroupedChart() {{
      const svg = document.getElementById("groupedChart");
      const w = 1220, h = 520, m = {{ top: 30, right: 24, bottom: 80, left: 92 }};
      const maxV = Math.max(...rawData.flatMap(item => [item.time_to_first_visible_char_ms, item.time_to_full_visible_answer_ms, item.server_retrieval_ms])) * 1.12;
      const {{ pw, ph }} = drawAxes(svg, w, h, m, maxV, 5, v => Math.round(v / 1000) + "k");
      const gw = pw / rawData.length, bw = Math.min(24, gw / 5), gap = 8;
      const series = [
        {{ key: "time_to_first_visible_char_ms", color: colors.first }},
        {{ key: "time_to_full_visible_answer_ms", color: colors.full }},
        {{ key: "server_retrieval_ms", color: colors.server }}
      ];
      rawData.forEach((item, idx) => {{
        const gx = m.left + gw * idx;
        const tbw = series.length * bw + (series.length - 1) * gap;
        const sx = gx + (gw - tbw) / 2;
        series.forEach((s, si) => {{
          const bh = (item[s.key] / maxV) * ph;
          svg.appendChild(makeSvg("rect", {{ x: sx + si * (bw + gap), y: m.top + ph - bh, width: bw, height: bh, rx: 6, fill: s.color }}));
        }});
        addText(svg, gx + gw / 2, h - 28, item.group, {{ "text-anchor": "middle", fill: colors.axis, "font-size": 13 }});
      }});
    }}

    function drawRatioChart() {{
      const svg = document.getElementById("ratioChart");
      const w = 1220, h = 420, m = {{ top: 24, right: 24, bottom: 76, left: 92 }};
      const maxV = Math.max(...rawData.map(item => item.ratio)) * 1.18;
      const {{ pw, ph }} = drawAxes(svg, w, h, m, maxV, 5, v => (v * 100).toFixed(0) + "%");
      const bs = pw / rawData.length, bw = Math.min(52, bs * 0.62);
      rawData.forEach((item, idx) => {{
        const bh = (item.ratio / maxV) * ph;
        const x = m.left + idx * bs + (bs - bw) / 2;
        const y = m.top + ph - bh;
        svg.appendChild(makeSvg("rect", {{ x, y, width: bw, height: bh, rx: 10, fill: colors.ratio }}));
        addText(svg, x + bw / 2, y - 8, fmtPct(item.ratio), {{ "text-anchor": "middle", fill: colors.text, "font-size": 12, "font-weight": 600 }});
        addText(svg, x + bw / 2, h - 28, item.group, {{ "text-anchor": "middle", fill: colors.axis, "font-size": 13 }});
      }});
    }}

    function drawAverageChart() {{
      const svg = document.getElementById("averageChart");
      const w = 1040, h = 460, m = {{ top: 26, right: 90, bottom: 74, left: 90 }};
      const pw = w - m.left - m.right, ph = h - m.top - m.bottom;
      const lMax = Math.max(average.first, average.full, average.server) * 1.16;
      const rMax = average.ratio * 1.4;
      for (let i = 0; i <= 5; i++) {{
        const lv = lMax * (i / 5), y = m.top + ph - (lv / lMax) * ph;
        svg.appendChild(makeSvg("line", {{ x1: m.left, y1: y, x2: m.left + pw, y2: y, stroke: colors.grid, "stroke-width": 1 }}));
        addText(svg, m.left - 10, y + 4, Math.round(lv / 1000) + "k", {{ "text-anchor": "end", fill: colors.axis, "font-size": 12 }});
        const rv = rMax * (i / 5);
        addText(svg, w - m.right + 10, y + 4, (rv * 100).toFixed(0) + "%", {{ "text-anchor": "start", fill: colors.axis, "font-size": 12 }});
      }}
      svg.appendChild(makeSvg("line", {{ x1: m.left, y1: m.top, x2: m.left, y2: m.top + ph, stroke: colors.axis, "stroke-width": 1.2 }}));
      svg.appendChild(makeSvg("line", {{ x1: w - m.right, y1: m.top, x2: w - m.right, y2: m.top + ph, stroke: colors.axis, "stroke-width": 1.2 }}));
      svg.appendChild(makeSvg("line", {{ x1: m.left, y1: m.top + ph, x2: w - m.right, y2: m.top + ph, stroke: colors.axis, "stroke-width": 1.2 }}));
      const items = [
        {{ label: "首字可见", value: average.first, color: colors.first, axis: "left", text: fmtMs(average.first) }},
        {{ label: "完整答案", value: average.full, color: colors.full, axis: "left", text: fmtMs(average.full) }},
        {{ label: "服务端检索", value: average.server, color: colors.server, axis: "left", text: fmtMs(average.server) }},
        {{ label: "检索占比", value: average.ratio, color: colors.ratio, axis: "right", text: fmtPct(average.ratio) }}
      ];
      const slot = pw / items.length, bw = Math.min(110, slot * 0.44);
      items.forEach((item, idx) => {{
        const x = m.left + slot * idx + (slot - bw) / 2;
        const sh = item.axis === "left" ? (item.value / lMax) * ph : (item.value / rMax) * ph;
        svg.appendChild(makeSvg("rect", {{ x, y: m.top + ph - sh, width: bw, height: sh, rx: 12, fill: item.color }}));
        addText(svg, x + bw / 2, m.top + ph - sh - 10, item.text, {{ "text-anchor": "middle", fill: colors.text, "font-size": 13, "font-weight": 700 }});
        addText(svg, x + bw / 2, h - 30, item.label, {{ "text-anchor": "middle", fill: colors.axis, "font-size": 14 }});
      }});
    }}

    function renderSummaryCards() {{
      const cards = [
        {{ label: "平均首字可见时间", value: fmtMs(average.first) }},
        {{ label: "平均完整答案时间", value: fmtMs(average.full) }},
        {{ label: "平均服务端检索时间", value: fmtMs(average.server) }},
        {{ label: "平均检索耗时占比", value: fmtPct(average.ratio) }}
      ];
      const container = document.getElementById("summaryCards");
      cards.forEach(card => {{
        const node = document.createElement("div");
        node.className = "stat-card";
        node.innerHTML = `<div class="stat-label">${{card.label}}</div><div class="stat-value">${{card.value}}</div>`;
        container.appendChild(node);
      }});
    }}

    function renderTable() {{
      const tbody = document.getElementById("detailRows");
      rawData.forEach(item => {{
        const row = document.createElement("tr");
        row.innerHTML = `<td class="question-cell" title="${{item.question}}">${{item.group}}: ${{item.question.slice(0, 40)}}...</td><td>${{fmtMs(item.time_to_first_visible_char_ms)}}</td><td>${{fmtMs(item.time_to_full_visible_answer_ms)}}</td><td>${{fmtMs(item.server_retrieval_ms)}}</td><td>${{fmtPct(item.ratio)}}</td>`;
        tbody.appendChild(row);
      }});
    }}

    function renderStageBreakdownLegend() {{
      const legend = document.getElementById("stageBreakdownLegend");
      stageMetricSeries.forEach(item => {{
        const node = document.createElement("span");
        node.innerHTML = `<i class="swatch" style="background: ${{item.color}};"></i> ${{item.label}}`;
        legend.appendChild(node);
      }});
    }}

    function drawStageBreakdownPieChart() {{
      const svg = document.getElementById("stageBreakdownPieChart");
      const summary = document.getElementById("stageBreakdownSummary");
      const w = 1220, h = 520;
      if (!stageBreakdownData.length || !stageBreakdownTotalMs) {{
        addText(svg, w / 2, h / 2, "当前样本里还没有可用的阶段耗时数据。", {{ "text-anchor": "middle", fill: colors.axis, "font-size": 18 }});
        summary.textContent = "提示：只有 meta 事件中包含 server_embed_ms / server_vector_search_ms / server_rerank_ms / server_prompt_build_ms 时，这里才会显示饼图。";
        return;
      }}
      const cx = 340, cy = 260, r = 150;
      let ca = 0;
      stageBreakdownData.forEach(item => {{
        const ratio = item.averageMs / stageBreakdownTotalMs;
        const sa = ratio * 360, ea = ca + sa;
        svg.appendChild(makeSvg("path", {{ d: describeArc(cx, cy, r, ca, ea), fill: item.color }}));
        if (ratio >= 0.06) {{
          const lp = polarToCartesian(cx, cy, r * 0.62, ca + sa / 2);
          addText(svg, lp.x, lp.y, (ratio * 100).toFixed(1) + "%", {{ "text-anchor": "middle", fill: "#fff", "font-size": 14, "font-weight": 700 }});
        }}
        ca = ea;
      }});
      const px = 620;
      addText(svg, px, 110, "平均阶段耗时", {{ fill: colors.text, "font-size": 24, "font-weight": 700 }});
      stageBreakdownData.forEach((item, idx) => {{
        const y = 170 + idx * 74, ratio = item.averageMs / stageBreakdownTotalMs;
        svg.appendChild(makeSvg("rect", {{ x: px, y: y - 16, width: 18, height: 18, rx: 4, fill: item.color }}));
        addText(svg, px + 30, y - 1, item.label, {{ fill: colors.text, "font-size": 16, "font-weight": 600 }});
        addText(svg, px + 260, y - 1, fmtMs(item.averageMs) + " / " + fmtPct(ratio), {{ fill: colors.axis, "font-size": 15, "text-anchor": "end" }});
      }});
      summary.textContent = "四个阶段的平均总耗时为 " + fmtMs(stageBreakdownTotalMs) + "，饼图展示的是它们在"生成前准备"中的平均时间占比。";
    }}

    renderSummaryCards();
    drawGroupedChart();
    drawRatioChart();
    drawAverageChart();
    renderTable();
    renderStageBreakdownLegend();
    drawStageBreakdownPieChart();
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\n报告已生成: {output_path.resolve()}")


def main() -> None:
    random.seed(42)
    all_questions = load_questions()
    selected = random.sample(all_questions, 5)

    print(f"随机选取 5 个问题:")
    for q in selected:
        print(f"  {q['question_id']}: {q['question']}")
    print()

    samples = run_benchmark(selected, top_k=10)

    output_path = REPO_ROOT / "evaluation" / "outputs" / "latency_benchmark_topk10.html"
    generate_html(samples, output_path)

    results_path = REPO_ROOT / "outputs" / "latency_benchmark_topk10.json"
    results_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"原始数据已保存: {results_path.resolve()}")


if __name__ == "__main__":
    main()
