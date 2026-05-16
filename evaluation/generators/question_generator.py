from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(slots=True)
class TestQuestion:
    id: str
    question: str
    relevant_chunk_ids: list[str]
    reference_answer: str
    category: str
    difficulty: str
    source_doc_title: str


QUESTION_GENERATION_PROMPT = """你是一个微信小程序开发文档测试专家。请基于以下文档内容，生成一个开发者可能会遇到的技术问题，并提供一个简明的参考答案。

文档标题：{title}
文档章节：{sections}
文档内容：
{content}

要求：
1. 问题应该是具体的、可检索的技术问题（不是概念解释题）
2. 参考答案应该准确、简洁（100-300字）
3. 问题难度标记为 easy/medium/hard 之一，根据问题的复杂程度

请只输出 JSON 格式（不要包含其他文字）：
{{"question": "生成的问题", "answer": "参考答案", "difficulty": "easy|medium|hard"}}"""


def generate_questions(
    chunks_jsonl: Path,
    generator_client,
    num_questions: int = 50,
    output_path: Path | None = None,
    random_seed: int = 42,
) -> list[TestQuestion]:
    from embedding_indexing.io import iter_chunks

    random.seed(random_seed)

    all_chunks = list(iter_chunks(chunks_jsonl))
    text_chunks = [c for c in all_chunks if c.chunk_type == "text"]

    category_map = _group_by_category(text_chunks)
    sampled: list[tuple[str, object]] = []
    for category, chunks in category_map.items():
        count = max(1, int(num_questions * len(chunks) / len(text_chunks)))
        sampled_chunks = random.sample(chunks, min(count, len(chunks)))
        for chunk_info in sampled_chunks:
            sampled.append((category, chunk_info))

    random.shuffle(sampled)
    sampled = sampled[:num_questions]

    questions: list[TestQuestion] = []
    _save_path = output_path  # capture for incremental saving
    for idx, (category, chunk_info) in enumerate(tqdm(sampled, desc="Generating questions")):
        chunk = chunk_info
        title = chunk.title.strip("# ")
        sections = " > ".join(chunk.section_path) if chunk.section_path else title
        content = chunk.chunk_text[:1200]

        prompt = QUESTION_GENERATION_PROMPT.format(
            title=title,
            sections=sections,
            content=content,
        )

        try:
            from json import JSONDecodeError

            response = _call_llm(generator_client, prompt)
            data = _extract_json(response)
            if data is None:
                continue

            question_text = str(data.get("question", "")).strip()
            answer_text = str(data.get("answer", "")).strip()
            difficulty = str(data.get("difficulty", "medium")).strip().lower()

            if not question_text or not answer_text:
                continue

            q = TestQuestion(
                id=f"q{idx + 1:03d}",
                question=question_text,
                relevant_chunk_ids=[chunk.chunk_id],
                reference_answer=answer_text,
                category=category,
                difficulty=difficulty if difficulty in ("easy", "medium", "hard") else "medium",
                source_doc_title=title,
            )
            questions.append(q)

        except Exception:
            continue

        # Incremental save every 5 questions
        if _save_path and len(questions) % 5 == 0:
            _save_questions_incremental(_save_path, questions, all_chunks, generator_client)

        time.sleep(0.3)

    random.shuffle(questions)
    for i, q in enumerate(questions):
        q.id = f"q{i + 1:03d}"

    if output_path:
        _save_questions_incremental(output_path, questions, all_chunks, generator_client)

    return questions


def load_questions(path: Path) -> list[TestQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TestQuestion(
            id=q["id"],
            question=q["question"],
            relevant_chunk_ids=q["relevant_chunk_ids"],
            reference_answer=q.get("reference_answer", ""),
            category=q.get("category", "general"),
            difficulty=q.get("difficulty", "medium"),
            source_doc_title=q.get("source_doc_title", ""),
        )
        for q in data["questions"]
    ]


def _save_questions_incremental(path: Path, questions: list[TestQuestion], all_chunks: list, generator_client) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "meta": {
            "dataset_name": "wechat-miniprogram-eval-v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_documents": len(set(c.doc_id for c in all_chunks)),
            "total_chunks": len(all_chunks),
            "total_questions": len(questions),
            "generation_model": getattr(generator_client, "model", "unknown"),
        },
        "questions": [
            {
                "id": q.id,
                "question": q.question,
                "relevant_chunk_ids": q.relevant_chunk_ids,
                "reference_answer": q.reference_answer,
                "category": q.category,
                "difficulty": q.difficulty,
                "source_doc_title": q.source_doc_title,
            }
            for q in questions
        ],
    }
    path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_by_category(chunks: list) -> dict[str, list]:
    categories: dict[str, list] = {}
    for chunk in chunks:
        nav = chunk.nav_path
        if nav:
            cat = nav[0]
        else:
            cat = "other"
        categories.setdefault(cat, []).append(chunk)
    return categories


def _call_llm(client, prompt: str) -> str:
    from llm.generator import _local_service_proxy_guard

    normalized_host = _normalize_host(client.host)
    local_host = _is_local(normalized_host)

    with _local_service_proxy_guard(enabled=local_host):
        import ollama

    client_kwargs = {"host": normalized_host, "timeout": client.timeout}
    if local_host:
        client_kwargs["trust_env"] = False
    with _local_service_proxy_guard(enabled=local_host):
        oc = ollama.Client(**client_kwargs)

    result = oc.generate(model=client.model, prompt=prompt, stream=False, think=False)
    return str(result.get("response", ""))


def _normalize_host(host: str) -> str:
    from llm.networking import normalize_local_service_url

    return normalize_local_service_url(host) or host


def _is_local(host: str) -> bool:
    from llm.networking import is_local_service_url

    return is_local_service_url(host)


def _extract_json(text: str) -> dict | None:
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
