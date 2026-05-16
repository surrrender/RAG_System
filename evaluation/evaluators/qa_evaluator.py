from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from tqdm import tqdm

from evaluation.evaluators.metrics import compute_bleu, compute_rouge
from evaluation.generators.question_generator import TestQuestion, _call_llm


@dataclass(slots=True)
class PerQuestionQAResult:
    question_id: str
    question: str
    category: str
    difficulty: str
    reference_answer: str
    generated_answer: str
    bleu_1: float
    bleu_2: float
    bleu_4: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    faithfulness_score: int
    relevance_score: int
    completeness_score: int
    accuracy_score: int
    overall_score: float
    retrieval_count: int
    model: str


@dataclass(slots=True)
class QAEvalResult:
    questions: list[PerQuestionQAResult]
    avg_bleu_1: float
    avg_bleu_2: float
    avg_bleu_4: float
    avg_rouge_1_f: float
    avg_rouge_2_f: float
    avg_rouge_l_f: float
    avg_faithfulness: float
    avg_relevance: float
    avg_completeness: float
    avg_accuracy: float
    avg_overall: float

    bleu_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    rouge_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    human_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)


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


def evaluate_qa(
    service,
    generator_client,
    questions: list[TestQuestion],
) -> QAEvalResult:
    results: list[PerQuestionQAResult] = []

    for q in tqdm(questions, desc="QA evaluation"):
        try:
            answer_result = service.answer_question(q.question, top_k=5)
            generated_answer = answer_result.answer
            retrieval_count = answer_result.retrieval_count
            model = answer_result.model

            # Collect context for faithfulness evaluation
            context_parts: list[str] = []
            for citation in answer_result.citations:
                text = citation.text or ""
                if text:
                    context_parts.append(f"[{citation.chunk_id}] {text[:800]}")
            context = "\n\n".join(context_parts) if context_parts else "（无检索结果）"
        except Exception:
            generated_answer = "（生成失败）"
            retrieval_count = 0
            model = "unknown"
            context = "（无检索结果）"

        bleu = compute_bleu([q.reference_answer], generated_answer)
        rouge = compute_rouge([q.reference_answer], generated_answer)

        faithfulness = 3
        relevance = 3
        completeness = 3
        accuracy = 3
        overall = 3.0

        if generated_answer != "（生成失败）":
            try:
                f_prompt = FAITHFULNESS_PROMPT.format(context=context, answer=generated_answer)
                f_response = _call_llm(generator_client, f_prompt)
                f_data = _extract_json(f_response)
                if f_data:
                    faithfulness = int(f_data.get("score", 3))
                time.sleep(0.3)
            except Exception:
                pass

            try:
                h_prompt = HUMAN_SCORE_PROMPT.format(
                    question=q.question,
                    reference=q.reference_answer,
                    answer=generated_answer,
                )
                h_response = _call_llm(generator_client, h_prompt)
                h_data = _extract_json(h_response)
                if h_data:
                    relevance = int(h_data.get("relevance", 3))
                    completeness = int(h_data.get("completeness", 3))
                    accuracy = int(h_data.get("accuracy", 3))
                    overall = float(h_data.get("overall", 3.0))
                time.sleep(0.3)
            except Exception:
                pass

        results.append(
            PerQuestionQAResult(
                question_id=q.id,
                question=q.question,
                category=q.category,
                difficulty=q.difficulty,
                reference_answer=q.reference_answer,
                generated_answer=generated_answer,
                bleu_1=bleu["bleu_1"],
                bleu_2=bleu["bleu_2"],
                bleu_4=bleu["bleu_4"],
                rouge_1_f=rouge["rouge_1_f"],
                rouge_2_f=rouge["rouge_2_f"],
                rouge_l_f=rouge["rouge_l_f"],
                faithfulness_score=faithfulness,
                relevance_score=relevance,
                completeness_score=completeness,
                accuracy_score=accuracy,
                overall_score=overall,
                retrieval_count=retrieval_count,
                model=model,
            )
        )

    n = max(len(results), 1)
    summary = QAEvalResult(
        questions=results,
        avg_bleu_1=sum(r.bleu_1 for r in results) / n,
        avg_bleu_2=sum(r.bleu_2 for r in results) / n,
        avg_bleu_4=sum(r.bleu_4 for r in results) / n,
        avg_rouge_1_f=sum(r.rouge_1_f for r in results) / n,
        avg_rouge_2_f=sum(r.rouge_2_f for r in results) / n,
        avg_rouge_l_f=sum(r.rouge_l_f for r in results) / n,
        avg_faithfulness=sum(r.faithfulness_score for r in results) / n,
        avg_relevance=sum(r.relevance_score for r in results) / n,
        avg_completeness=sum(r.completeness_score for r in results) / n,
        avg_accuracy=sum(r.accuracy_score for r in results) / n,
        avg_overall=sum(r.overall_score for r in results) / n,
    )

    _compute_groups(results, summary)
    return summary


def _compute_groups(results: list[PerQuestionQAResult], summary: QAEvalResult) -> None:
    groups: dict[str, list[PerQuestionQAResult]] = {}
    for r in results:
        groups.setdefault(r.difficulty, []).append(r)

    for key, group_results in groups.items():
        n = max(len(group_results), 1)
        summary.bleu_by_difficulty[key] = {
            "bleu_1": sum(r.bleu_1 for r in group_results) / n,
            "bleu_2": sum(r.bleu_2 for r in group_results) / n,
            "bleu_4": sum(r.bleu_4 for r in group_results) / n,
        }
        summary.rouge_by_difficulty[key] = {
            "rouge_1_f": sum(r.rouge_1_f for r in group_results) / n,
            "rouge_2_f": sum(r.rouge_2_f for r in group_results) / n,
            "rouge_l_f": sum(r.rouge_l_f for r in group_results) / n,
        }
        summary.human_by_difficulty[key] = {
            "faithfulness": sum(r.faithfulness_score for r in group_results) / n,
            "relevance": sum(r.relevance_score for r in group_results) / n,
            "completeness": sum(r.completeness_score for r in group_results) / n,
            "accuracy": sum(r.accuracy_score for r in group_results) / n,
            "overall": sum(r.overall_score for r in group_results) / n,
        }


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
