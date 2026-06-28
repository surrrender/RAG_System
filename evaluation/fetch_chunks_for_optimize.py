"""
Fetch top-10 chunks for all 40 questions (with & without reranker) and save raw data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm.config import load_settings
from llm.retrieval import Retriever

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "data" / "manual_questions.json"
OUTPUT_DIR = EVAL_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def retrieve(questions, retriever, top_k=10):
    results = []
    for idx, q in enumerate(questions):
        qid = q["id"]
        print(f"  [{idx+1}/40] {qid}: {q['question'][:60]}...")
        try:
            chunks, metrics = retriever.retrieve_with_metrics(q["question"], top_k=top_k)
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
        except Exception as exc:
            print(f"    ERROR: {exc}")
            retrieved = []
        results.append({
            "question_id": qid,
            "question": q["question"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", "medium"),
            "retrieved_chunks": retrieved,
        })
    return results

def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    print(f"Loaded {len(questions)} questions\n")

    settings = load_settings()

    # Build reranker directly so we control the env & it's fully from cache
    import os as _os
    from embedding_indexing.rerankers import build_reranker as _build_reranker

    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
        _os.environ.pop(k, None)
    _os.environ["HF_HUB_OFFLINE"] = "1"
    _os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        reranker = _build_reranker(
            provider=settings.reranker_provider,
            model_name=settings.reranker_model,
            offline=True,
            device=settings.reranker_device,
        )
        print("  Reranker loaded from cache successfully")
    except Exception as exc:
        print(f"  Reranker unavailable: {exc}")
        reranker = None

    all_data = {}

    for mode, use_reranker in [("with_reranker", True), ("no_reranker", False)]:
        print(f"\n=== Mode: {mode} (reranker={use_reranker}) ===")
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
            disable_reranker=not use_reranker,
        )
        # Inject the pre-loaded reranker so warm_up doesn't try to load again
        if use_reranker and reranker is not None:
            retriever._reranker = reranker

        print("  Warming up...")
        retriever.warm_up()
        print("  Done. Retrieving...")
        all_data[mode] = retrieve(questions, retriever, top_k=10)

    raw_path = OUTPUT_DIR / "_raw_chunks_for_annotation.json"
    raw_path.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw data saved to {raw_path}")

if __name__ == "__main__":
    main()
