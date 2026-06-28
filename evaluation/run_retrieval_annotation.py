"""
检索召回标注工具
================
用法:
  不启用 reranker:
    python -m evaluation.run_retrieval_annotation --no-reranker -o outputs/annotation_no_reranker.html

  启用 reranker:
    python -m evaluation.run_retrieval_annotation --use-reranker -o outputs/annotation_with_reranker.html

流程：
  1. 加载 manual_questions.json 中的 40 个问题
  2. 创建 Retriever，对每个问题检索 top-10 chunk
  3. 生成独立的 HTML 交互页面，供人工标注相关 chunk
  4. 在浏览器中打开 HTML，逐题标注，最后导出 JSON 结果
  5. 导出结果后用 evaluate_annotation.py 计算指标
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm.config import load_settings
from llm.retrieval import Retriever

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "data" / "manual_questions.json"
OUTPUT_DIR = EVAL_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_annotation_html(
    question_data: list[dict],
    reranker_enabled: bool,
    top_k: int,
    output_path: Path,
) -> None:
    import json as _json

    question_data_json = _json.dumps(question_data, ensure_ascii=False, indent=2)
    config_label = "启用 Reranker" if reranker_enabled else "未启用 Reranker"
    file_suffix = "with_reranker" if reranker_enabled else "no_reranker"
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    total = len(question_data)
    reranker_enabled_json = _json.dumps(reranker_enabled)
    header_config = f"top-{top_k} · {config_label} · 共 {total} 题"

    html = _HTML_BEFORE_DATA + question_data_json + _HTML_AFTER_DATA
    html = html.replace("__HEADER_CONFIG__", header_config)
    html = html.replace("__TOTAL__", str(total))
    html = html.replace("__RERANKER_ENABLED__", reranker_enabled_json)
    html = html.replace("__TOP_K__", str(top_k))
    html = html.replace("__GENERATED_AT__", generated_at)
    html = html.replace("__FILE_SUFFIX__", file_suffix)

    output_path.write_text(html, encoding="utf-8")
    print(f"\n标注页面已生成: {output_path.resolve()}")
    print(f"请在浏览器中打开此文件，逐题标注相关 chunk，完成后点击「导出结果」下载 JSON。")
    print(f"提示: 支持键盘 ← → 切换题目，点击 chunk 卡片切换选中状态。")


_HTML_BEFORE_DATA = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>检索召回标注工具</title>
<style>
  :root {
    --primary: #4f46e5;
    --primary-light: #eef2ff;
    --bg: #f8fafc;
    --card-bg: #fff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
    --success: #22c55e;
    --success-bg: #f0fdf4;
    --success-border: #bbf7d0;
    --danger: #ef4444;
    --danger-bg: #fef2f2;
    --radius: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    min-height: 100vh;
  }
  .header {
    background: var(--primary);
    color: #fff; padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3);
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .config-badge {
    background: rgba(255,255,255,0.2);
    padding: 4px 12px; border-radius: 20px;
    font-size: 13px;
  }
  .container { max-width: 900px; margin: 0 auto; padding: 24px 16px; }
  .nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px; gap: 12px;
  }
  .nav-btn {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 10px 20px;
    cursor: pointer; font-size: 14px; color: var(--text);
    transition: all 0.15s; display: flex; align-items: center; gap: 6px;
  }
  .nav-btn:hover:not(:disabled) { border-color: var(--primary); color: var(--primary); }
  .nav-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .nav-progress { font-size: 14px; color: var(--text-muted); white-space: nowrap; }
  .nav-progress strong { color: var(--text); }
  .progress-bar-wrapper {
    flex: 1; height: 6px; background: var(--border); border-radius: 3px;
    overflow: hidden; max-width: 200px;
  }
  .progress-bar-fill {
    height: 100%; background: var(--primary); border-radius: 3px;
    transition: width 0.3s;
  }
  .question-card {
    background: var(--card-bg);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 20px 24px; margin-bottom: 20px; box-shadow: var(--shadow);
  }
  .question-card .q-label {
    font-size: 12px; color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px;
  }
  .question-card .q-text {
    font-size: 17px; font-weight: 600; line-height: 1.5;
  }
  .question-card .q-meta { margin-top: 12px; display: flex; gap: 12px; flex-wrap: wrap; }
  .question-card .q-tag {
    font-size: 12px; padding: 2px 10px; border-radius: 12px;
    background: var(--primary-light); color: var(--primary);
  }
  .chunk-list { display: flex; flex-direction: column; gap: 12px; }
  .chunk-card {
    background: var(--card-bg);
    border: 2px solid var(--border); border-radius: var(--radius);
    padding: 16px 20px; box-shadow: var(--shadow);
    cursor: pointer; transition: all 0.15s;
    position: relative;
  }
  .chunk-card:hover { border-color: #cbd5e1; }
  .chunk-card.selected { border-color: var(--success); background: var(--success-bg); }
  .chunk-card .chunk-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
  }
  .chunk-card .chunk-rank {
    font-size: 13px; font-weight: 600; color: var(--text-muted);
    min-width: 60px;
  }
  .chunk-card .chunk-score {
    font-size: 13px; color: var(--text-muted);
    background: #f1f5f9; padding: 2px 10px; border-radius: 10px;
  }
  .chunk-card .chunk-title {
    font-size: 14px; font-weight: 500; color: var(--text);
    margin-bottom: 4px;
  }
  .chunk-card .chunk-section {
    font-size: 12px; color: var(--text-muted); margin-bottom: 8px;
  }
  .chunk-card .chunk-text {
    font-size: 14px; color: var(--text); line-height: 1.7;
    max-height: 120px; overflow-y: auto;
    padding: 8px 12px; background: #f8fafc;
    border-radius: 4px; border: 1px solid #f1f5f9;
    white-space: pre-wrap; word-break: break-all;
  }
  .chunk-card .chunk-text.expanded { max-height: none; }
  .chunk-card .toggle-expand {
    font-size: 12px; color: var(--primary); cursor: pointer;
    margin-top: 4px; display: inline-block;
  }
  .chunk-card .chunk-status {
    position: absolute; top: 16px; right: 16px;
    width: 24px; height: 24px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700;
  }
  .chunk-card .chunk-status.relevant { background: var(--success); color: #fff; }
  .chunk-card .chunk-status.irrelevant { background: var(--border); color: var(--text-muted); }
  .chunk-card .chunk-id-display {
    font-size: 11px; color: #94a3b8; margin-top: 4px;
    font-family: "SF Mono", "Cascadia Code", monospace;
  }
  .chunk-card.selected .chunk-rank { color: var(--success); }
  .chunk-card.selected .chunk-score { background: var(--success-border); color: #15803d; }
  .action-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 24px; padding-top: 16px;
    border-top: 1px solid var(--border);
  }
  .action-bar .stats { font-size: 14px; color: var(--text-muted); }
  .action-bar .stats strong { color: var(--text); }
  .btn-export {
    background: var(--primary); color: #fff; border: none;
    padding: 10px 28px; border-radius: var(--radius); font-size: 14px;
    cursor: pointer; transition: background 0.15s; font-weight: 500;
  }
  .btn-export:hover { background: #4338ca; }
  .btn-reset {
    background: transparent; color: var(--text-muted); border: 1px solid var(--border);
    padding: 10px 20px; border-radius: var(--radius); font-size: 14px;
    cursor: pointer; transition: all 0.15s;
  }
  .btn-reset:hover { border-color: var(--danger); color: var(--danger); }
  .toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #1e293b; color: #fff; padding: 12px 24px;
    border-radius: var(--radius); font-size: 14px;
    opacity: 0; transition: opacity 0.3s; pointer-events: none;
    z-index: 200;
  }
  .toast.show { opacity: 1; }
  @media (max-width: 640px) {
    .header h1 { font-size: 15px; }
    .container { padding: 16px 8px; }
    .question-card { padding: 16px; }
    .chunk-card { padding: 12px 16px; }
    .nav-bar { flex-wrap: wrap; }
    .progress-bar-wrapper { max-width: 100%; flex-basis: 100%; order: -1; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>Retrieval Annotation Tool</h1>
  <span class="config-badge">__HEADER_CONFIG__</span>
</div>

<div class="container">
  <div class="nav-bar">
    <button class="nav-btn" id="prevBtn" onclick="navigate(-1)">&larr; Prev</button>
    <span class="nav-progress">Question <strong id="currentIdx">1</strong> / __TOTAL__</span>
    <div class="progress-bar-wrapper">
      <div class="progress-bar-fill" id="progressFill" style="width:0%"></div>
    </div>
    <button class="nav-btn" id="nextBtn" onclick="navigate(1)">Next &rarr;</button>
  </div>

  <div class="question-card">
    <div class="q-label">Question <span id="qNumber">1</span></div>
    <div class="q-text" id="qText"></div>
    <div class="q-meta">
      <span class="q-tag" id="qCategory"></span>
      <span class="q-tag" id="qDifficulty"></span>
    </div>
  </div>

  <div class="chunk-list" id="chunkList"></div>
  <div id="noChunksMsg" style="display:none;text-align:center;padding:40px;color:var(--text-muted);font-size:15px;">
    No chunks retrieved for this question.
  </div>

  <div class="action-bar">
    <div class="stats">
      Annotated <strong id="annotatedCount">0</strong> / __TOTAL__ &middot;
      Selected <strong id="selectedCount">0</strong> chunks
    </div>
    <div style="display:flex;gap:8px">
      <button class="btn-reset" onclick="resetCurrent()">Reset</button>
      <button class="btn-export" onclick="exportResults()">Export Results</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
var QUESTIONS = """


_HTML_AFTER_DATA = """;

var currentIndex = 0;
var annotations = {};
var lastSaved = '';

function loadState() {
  try {
    var saved = localStorage.getItem('retrieval_annotations');
    if (saved) { annotations = JSON.parse(saved); lastSaved = saved; }
  } catch(e) {}
}

function saveState() {
  try {
    var s = JSON.stringify(annotations);
    localStorage.setItem('retrieval_annotations', s);
    lastSaved = s;
  } catch(e) {}
}

function renderQuestion() {
  var q = QUESTIONS[currentIndex];
  var qid = q.id;
  document.getElementById('currentIdx').textContent = currentIndex + 1;
  document.getElementById('qNumber').textContent = currentIndex + 1;
  document.getElementById('progressFill').style.width = ((currentIndex + 1) / QUESTIONS.length * 100) + '%';
  document.getElementById('qText').textContent = q.question;
  document.getElementById('qCategory').textContent = q.category;
  document.getElementById('qDifficulty').textContent = q.difficulty;
  document.getElementById('prevBtn').disabled = currentIndex === 0;
  document.getElementById('nextBtn').disabled = currentIndex === QUESTIONS.length - 1;

  var container = document.getElementById('chunkList');
  container.innerHTML = '';
  var selected = annotations[qid] || [];
  var noMsg = document.getElementById('noChunksMsg');

  if (!q.chunks || q.chunks.length === 0) {
    noMsg.style.display = 'block';
    updateStats();
    return;
  }
  noMsg.style.display = 'none';

  q.chunks.forEach(function(chunk, i) {
    var isSelected = selected.indexOf(chunk.chunk_id) !== -1;
    var textId = 'chunkText_' + i;
    var card = document.createElement('div');
    card.className = 'chunk-card' + (isSelected ? ' selected' : '');
    card.setAttribute('data-chunk-id', chunk.chunk_id);
    card.onclick = function(e) {
      if (e.target.tagName === 'A' || e.target.classList.contains('toggle-expand')) return;
      toggleChunk(chunk.chunk_id, qid);
    };
    var sectionStr = chunk.section_path && chunk.section_path.length
      ? chunk.section_path.join(' \u203a ') : '';
    card.innerHTML =
      '<div class="chunk-header">' +
        '<span class="chunk-rank">#' + (i + 1) + '</span>' +
        '<span class="chunk-score">score: ' + chunk.score.toFixed(4) + '</span>' +
        '<div class="chunk-status ' + (isSelected ? 'relevant' : 'irrelevant') + '">' +
          (isSelected ? '\u2713' : '') +
        '</div>' +
      '</div>' +
      (chunk.title ? '<div class="chunk-title">' + escapeHtml(chunk.title) + '</div>' : '') +
      (sectionStr ? '<div class="chunk-section">' + escapeHtml(sectionStr) + '</div>' : '') +
      '<div class="chunk-text" id="' + textId + '">' + escapeHtml(chunk.text) + '</div>' +
      '<a class="toggle-expand" onclick="toggleTextExpand(this)">Expand</a>' +
      '<div class="chunk-id-display">' + chunk.chunk_id + '</div>';
    container.appendChild(card);
  });
  updateStats();
}

function toggleChunk(chunkId, qid) {
  if (!annotations[qid]) annotations[qid] = [];
  var idx = annotations[qid].indexOf(chunkId);
  if (idx === -1) { annotations[qid].push(chunkId); }
  else { annotations[qid].splice(idx, 1); }
  saveState();
  renderQuestion();
}

function toggleTextExpand(link) {
  var el = link.parentElement.querySelector('.chunk-text');
  if (el) {
    el.classList.toggle('expanded');
    link.textContent = el.classList.contains('expanded') ? 'Collapse' : 'Expand';
  }
}

function resetCurrent() {
  var q = QUESTIONS[currentIndex];
  annotations[q.id] = [];
  saveState();
  renderQuestion();
  showToast('Reset annotations for this question');
}

function navigate(delta) {
  var next = currentIndex + delta;
  if (next < 0 || next >= QUESTIONS.length) return;
  currentIndex = next;
  renderQuestion();
}

function updateStats() {
  var totalAnnotated = 0, totalSelected = 0;
  QUESTIONS.forEach(function(q) {
    var sel = annotations[q.id] || [];
    if (sel.length > 0) totalAnnotated++;
    totalSelected += sel.length;
  });
  document.getElementById('annotatedCount').textContent = totalAnnotated;
  document.getElementById('selectedCount').textContent = totalSelected;
}

function showToast(msg) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(function() { t.classList.remove('show'); }, 2000);
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function exportResults() {
  var result = {
    config: {
      reranker_enabled: __RERANKER_ENABLED__,
      top_k: __TOP_K__,
      generated_at: '__GENERATED_AT__',
    },
    annotations: []
  };
  QUESTIONS.forEach(function(q) {
    result.annotations.push({
      question_id: q.id,
      question: q.question,
      category: q.category,
      difficulty: q.difficulty,
      relevant_chunk_ids: annotations[q.id] || [],
      all_retrieved_chunks: q.chunks.map(function(c) {
        return {
          chunk_id: c.chunk_id,
          score: c.score,
          title: c.title,
          section_path: c.section_path,
          rank: c.rank,
        };
      })
    });
  });
  var blob = new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'annotation_results___FILE_SUFFIX__.json';
  a.click();
  URL.revokeObjectURL(url);
  showToast('Exported ' + result.annotations.length + ' question annotations');
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
});

loadState();
renderQuestion();
</script>
</body>
</html>"""


def retrieve_chunks_for_questions(
    questions: list[dict],
    retriever: Retriever,
    top_k: int = 10,
) -> list[dict]:
    results = []
    total = len(questions)
    for idx, q in enumerate(questions):
        qid = q["id"]
        print(f"  [{idx + 1}/{total}] {qid}: {q['question'][:60]}...")
        try:
            chunks, metrics = retriever.retrieve_with_metrics(
                q["question"], top_k=top_k
            )
            retrieved = [
                {
                    "rank": i + 1,
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "title": c.title or "",
                    "section_path": c.section_path or [],
                    "text": c.text or "",
                }
                for i, c in enumerate(chunks[:top_k])
            ]
            embed_ms = float(getattr(metrics, "embed_ms", 0) or 0)
            search_ms = float(getattr(metrics, "vector_search_ms", 0) or 0)
            rerank_ms = float(getattr(metrics, "rerank_ms", 0) or 0)
        except Exception as exc:
            print(f"    ERROR retrieving chunks: {exc}")
            retrieved = []
            embed_ms = search_ms = rerank_ms = 0.0

        results.append({
            "id": qid,
            "question": q["question"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", "medium"),
            "chunks": retrieved,
            "timing_ms": {
                "embed": round(embed_ms, 1),
                "search": round(search_ms, 1),
                "rerank": round(rerank_ms, 1),
            },
        })
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="检索召回标注工具")
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="不启用 reranker（默认）",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="启用 reranker",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="每个问题检索的 chunk 数量（默认 10）",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出的 HTML 文件路径",
    )
    parser.add_argument(
        "-q", "--questions",
        type=str,
        default=str(QUESTIONS_FILE),
        help=f"问题 JSON 文件路径（默认 {QUESTIONS_FILE}）",
    )
    args = parser.parse_args()

    reranker_enabled = args.use_reranker
    if not reranker_enabled and not args.no_reranker:
        print("默认未启用 reranker（可通过 --use-reranker 启用）")
    elif reranker_enabled:
        print("已启用 reranker")

    questions_file = Path(args.questions)
    if not questions_file.exists():
        print(f"错误: 问题文件不存在: {questions_file}")
        sys.exit(1)

    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    questions_list = questions["questions"]
    print(f"已加载 {len(questions_list)} 个问题，来自: {questions_file}")

    # Create retriever
    print("\n初始化 Retriever...")
    settings = load_settings()
    retriever = Retriever(
        qdrant_path=settings.qdrant_path,
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
        disable_reranker=not reranker_enabled,
    )
    print("  加载模型中...")
    retriever.warm_up()
    print("  模型加载完成")

    # Retrieve chunks
    print(f"\n检索 top-{args.top_k} chunks（共 {len(questions_list)} 题）...")
    question_data = retrieve_chunks_for_questions(questions_list, retriever, top_k=args.top_k)

    # Generate HTML
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "with_reranker" if reranker_enabled else "no_reranker"
        output_path = OUTPUT_DIR / f"annotation_{suffix}.html"

    generate_annotation_html(question_data, reranker_enabled, args.top_k, output_path)


if __name__ == "__main__":
    main()
