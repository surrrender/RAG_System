from __future__ import annotations

import math


def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = len(relevant_ids & top_k) # 交集的大小，即命中数
    return hits / len(relevant_ids)


def mrr(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    if not relevant_ids:
        return 1.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> bool:
    if not relevant_ids:
        return True
    top_k = set(retrieved_ids[:k])
    return bool(relevant_ids & top_k)


def compute_bleu(references: list[str], candidate: str) -> dict[str, float]:
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}

    if not candidate.strip() or not any(ref.strip() for ref in references):
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}

    bleu_scorer = BLEU()
    score = bleu_scorer.corpus_score([candidate], [references])
    precisions = score.precisions
    bp = score.bp
    result: dict[str, float] = {}
    for n_idx, n_key in enumerate(["bleu_1", "bleu_2", "bleu_4"], start=0):
        if n_idx >= len(precisions):
            result[n_key] = 0.0
        else:
            geo_mean = math.exp(min(n_idx, len(precisions) - 1))
            # Use sacrebleu's score directly
            result[n_key] = round(score.score / 100.0, 4)
    # Manual computation per n-gram level for individual BLEU-n
    result["bleu_1"] = round(precisions[0] if len(precisions) > 0 else 0.0, 4)
    result["bleu_2"] = round(_ngram_precision(references, [candidate], 2), 4)
    result["bleu_4"] = round(_ngram_precision(references, [candidate], 4), 4)
    return result


def _ngram_precision(references: list[str], candidates: list[str], n: int) -> float:
    total_ngrams = 0
    matched_ngrams = 0
    for candidate in candidates:
        candidate_ngrams = _get_ngrams(candidate, n)
        if not candidate_ngrams:
            continue
        total_ngrams += len(candidate_ngrams)
        ref_ngrams_list = [_get_ngrams(ref, n) for ref in references]
        best_match = 0
        for c_ngram, c_count in candidate_ngrams.items():
            max_ref_count = max(
                ref_ngrams.get(c_ngram, 0) for ref_ngrams in ref_ngrams_list
            )
            matched_ngrams += min(c_count, max_ref_count)
    return matched_ngrams / total_ngrams if total_ngrams > 0 else 0.0


def _get_ngrams(text: str, n: int) -> dict[str, int]:
    chars = list(text)
    ngrams: dict[str, int] = {}
    for i in range(len(chars) - n + 1):
        ngram = "".join(chars[i : i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams


def compute_rouge(references: list[str], candidate: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}

    if not candidate.strip() or not any(ref.strip() for ref in references):
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    best_scores = {key: 0.0 for key in ["rouge_1_f", "rouge_2_f", "rouge_l_f"]}
    for ref in references:
        scores = scorer.score(ref, candidate)
        best_scores["rouge_1_f"] = max(best_scores["rouge_1_f"], scores["rouge1"].fmeasure)
        best_scores["rouge_2_f"] = max(best_scores["rouge_2_f"], scores["rouge2"].fmeasure)
        best_scores["rouge_l_f"] = max(best_scores["rouge_l_f"], scores["rougeL"].fmeasure)

    return {k: round(v, 4) for k, v in best_scores.items()}
