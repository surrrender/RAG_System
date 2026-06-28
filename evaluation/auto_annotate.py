"""
LLM 自动语义标注脚本
=====================
对 40 个问题，检索 top-10 chunk，由 LLM 根据语义严格判断相关性。

用法:
  python -m evaluation.auto_annotate --no-reranker -o outputs/annotation_results_no_reranker_optimize.json
  python -m evaluation.auto_annotate --use-reranker -o outputs/annotation_results_with_reranker_optimize.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm
from llm.config import load_settings
from llm.retrieval import Retriever

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "data" / "manual_questions.json"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RELEVANCE_PROMPT = """你是一个严格的技术文档相关性判断专家。你的任务是判断一段文档内容是否与用户问题相关。

【核心原则】
- 仅当文档内容**直接有助于回答该问题**时，才标记为相关。
- **仅仅出现关键词不意味着相关**。例如问题问"如何实现登录"，文档只说"登录是什么"而没有流程步骤，则不相关。
- **仅仅属于同一个主题不意味着相关**。必须看文档是否提供了回答问题所需的具体信息。
- **宁可漏判，不可误判**。不确定时标记为不相关。

【判断标准】
- 相关 (true)：该片段包含能够**直接回答**用户问题某一方面所需的信息。例如：
  * 问题问"有哪些配置项"，文档列举了具体配置项 → 相关
  * 问题问"如何实现"，文档给出了步骤或方法 → 相关
  * 问题问"A和B的区别"，文档对比了两者 → 相关
- 不相关 (false)：即使片段内容围绕同一主题，但并未提供回答问题所需的信息。
  * 只出现了问题中的关键词但没有实质性回答内容
  * 仅说了概念定义但问题问的是"怎么用"
  * 仅提到了无关的细节

问题：{question}

文档内容：
{chunk_text}

请输出 JSON (不要其他内容)：{{"relevant": true/false, "reason": "10字内的理由"}}"""


_FALLBACK_RELEVANCE_PROMPT = """问题：{question}

文档内容：
{chunk_text}

这段文档是否包含回答该问题所需的信息？仅输出JSON：{{"relevant": true/false, "reason": "简短的判断理由"}}"""


def call_llm(prompt: str, host: str = "http://127.0.0.1:11434", model: str = "qwen3:8b") -> str:
    import urllib.request
    import urllib.error

    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "")
    except Exception as exc:
        return f"ERROR: {exc}"


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def is_chunk_relevant(
    question: str,
    chunk_text: str,
    host: str,
    model: str,
    max_retries: int = 2,
) -> tuple[bool, str]:
    prompt = RELEVANCE_PROMPT.format(question=question, chunk_text=chunk_text[:2000])
    for attempt in range(max_retries + 1):
        response = call_llm(prompt, host=host, model=model)
        if response.startswith("ERROR:"):
            if attempt < max_retries:
                time.sleep(2)
                continue
            prompt = _FALLBACK_RELEVANCE_PROMPT.format(question=question, chunk_text=chunk_text[:2000])
            response = call_llm(prompt, host=host, model=model)
            if response.startswith("ERROR:"):
                return False, f"LLM call failed: {response}"
        data = extract_json(response)
        if data and "relevant" in data:
            return bool(data["relevant"]), data.get("reason", "")
        if attempt < max_retries:
            time.sleep(1)
    return False, "Failed to parse LLM response"


def retrieve_and_annotate(
    questions: list[dict],
    retriever: Retriever,
    reranker_enabled: bool,
    top_k: int = 10,
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_model: str = "qwen3:8b",
) -> dict:
    annotations = []
    total = len(questions)

    for idx, q in enumerate(tqdm(questions, desc="Retrieval + Annotation")):
        qid = q["id"]
        question = q["question"]

        try:
            chunks, metrics = retriever.retrieve_with_metrics(question, top_k=top_k)
        except Exception as exc:
            tqdm.write(f"  [{idx + 1}/{total}] {qid} 检索失败: {exc}")
            annotations.append({
                "question_id": qid,
                "question": question,
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", "medium"),
                "relevant_chunk_ids": [],
                "all_retrieved_chunks": [],
            })
            continue

        retrieved = []
        relevant_ids = []

        for i, c in enumerate(chunks[:top_k]):
            chunk_info = {
                "chunk_id": c.chunk_id,
                "score": c.score,
                "title": c.title or "",
                "section_path": c.section_path or [],
                "rank": i + 1,
            }
            retrieved.append(chunk_info)

            chunk_text = c.text or ""
            if not chunk_text.strip():
                continue

            relevant, reason = is_chunk_relevant(
                question, chunk_text,
                host=ollama_host, model=ollama_model,
            )
            if relevant:
                relevant_ids.append(c.chunk_id)
                tqdm.write(
                    f"  [{idx + 1}/{total}] {qid}  #{i + 1} score={c.score:.4f} "
                    f"→ RELEVANT ({reason})"
                )

        annotations.append({
            "question_id": qid,
            "question": question,
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", "medium"),
            "relevant_chunk_ids": relevant_ids,
            "all_retrieved_chunks": retrieved,
        })

    result = {
        "config": {
            "reranker_enabled": reranker_enabled,
            "top_k": top_k,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "annotation_method": "llm_semantic_selection_v2",
            "criteria": "strict semantic relevance by qwen3:8b (宁可少选不可误判)",
        },
        "annotations": annotations,
    }

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LLM 自动语义标注")
    parser.add_argument("--no-reranker", action="store_true", help="不启用 reranker")
    parser.add_argument("--use-reranker", action="store_true", help="启用 reranker")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="每个问题检索 chunk 数量")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("-q", "--questions", type=str, default=str(QUESTIONS_FILE), help="问题文件")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", type=str, default="qwen3:8b")
    args = parser.parse_args()

    reranker_enabled = args.use_reranker
    if not reranker_enabled and not args.no_reranker:
        print("默认未启用 reranker")
    elif reranker_enabled:
        print("已启用 reranker")

    questions_data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = questions_data["questions"]
    print(f"已加载 {len(questions)} 个问题")

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

    print(f"\n检索 + LLM 标注中（top-{args.top_k}, 共 {len(questions)} 题）...")
    result = retrieve_and_annotate(
        questions, retriever, reranker_enabled,
        top_k=args.top_k,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
    )

    # Count statistics
    total_annotated = len(result["annotations"])
    total_relevant = sum(
        len(a["relevant_chunk_ids"]) for a in result["annotations"]
    )
    questions_with_relevant = sum(
        1 for a in result["annotations"] if a["relevant_chunk_ids"]
    )
    print(f"\n标注完成: {total_annotated} 题, "
          f"共标记 {total_relevant} 个相关 chunk, "
          f"{questions_with_relevant} 题至少有一个相关 chunk")

    output_path = Path(args.output or OUTPUT_DIR / "annotation_results.json")
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已保存: {output_path.resolve()}")


if __name__ == "__main__":
    main()
