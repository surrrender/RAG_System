"""
检索所有 chunk 数据，供人工/LLM 做相关性标注
"""
from __future__ import annotations
import json, sys, time
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

questions_data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
questions = questions_data["questions"]
print(f"已加载 {len(questions)} 个问题")

settings = load_settings()

def fetch(reranker_enabled: bool, label: str) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  检索配置: {label}")
    print(f"{'='*60}")
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

    results = []
    for idx, q in enumerate(tqdm(questions, desc=f"Retrieval ({label})")):
        qid = q["id"]
        try:
            chunks, metrics = retriever.retrieve_with_metrics(q["question"], top_k=10)
        except Exception as exc:
            print(f"  [{idx+1}] {qid} ERROR: {exc}")
            chunks = []

        retrieved = []
        for i, c in enumerate(chunks[:10]):
            retrieved.append({
                "chunk_id": c.chunk_id,
                "score": round(c.score, 6),
                "title": c.title or "",
                "section_path": c.section_path or [],
                "rank": i + 1,
                "text": c.text or "",
            })

        results.append({
            "question_id": qid,
            "question": q["question"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", "medium"),
            "retrieved_chunks": retrieved,
        })
    return results

# Fetch both configs
no_reranker_data = fetch(False, "无 Reranker")
with_reranker_data = fetch(True, "有 Reranker")

# Save combined data
combined = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "no_reranker": no_reranker_data,
    "with_reranker": with_reranker_data,
}
out = OUTPUT_DIR / "_raw_chunks_for_annotation.json"
out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n原始 chunk 数据已保存: {out}")
