from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class BM25Result:
    chunk_id: str
    score: float
    title: str | None = None
    url: str | None = None
    section_path: list[str] | None = None
    text: str | None = None


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[dict] = []
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._total_docs: int = 0

    def index_chunks(self, chunks: list[dict]) -> None:
        self._corpus = chunks
        self._total_docs = len(chunks)
        total_len = 0

        for chunk in chunks:
            tokens = _tokenize(chunk["text"])
            chunk["_tokens"] = tokens
            chunk["_tf"] = self._compute_tf(tokens)
            dl = len(tokens)
            self._doc_lengths.append(dl)
            total_len += dl
            for token in set(tokens):
                self._doc_freq[token] += 1

        self._avg_dl = total_len / max(self._total_docs, 1)

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for doc_idx, chunk in enumerate(self._corpus):
            score = self._score(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        return [
            BM25Result(
                chunk_id=self._corpus[idx]["chunk_id"],
                score=score,
                title=self._corpus[idx].get("title"),
                url=self._corpus[idx].get("url"),
                section_path=self._corpus[idx].get("section_path"),
                text=self._corpus[idx].get("text"),
            )
            for idx, score in top
        ]

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        tf_map = self._corpus[doc_idx]["_tf"]
        dl = self._doc_lengths[doc_idx]

        for token in query_tokens:
            df = self._doc_freq.get(token, 0)
            if df == 0:
                continue
            tf = tf_map.get(token, 0)
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * numerator / denominator

        return score

    @staticmethod
    def _compute_tf(tokens: list[str]) -> dict[str, int]:
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        return tf


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    for ch in text:
        if ch.isascii() and (ch.isalpha() or ch.isdigit()):
            buffer.append(ch.lower())
        else:
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            if ch.strip():
                tokens.append(ch)
    if buffer:
        tokens.append("".join(buffer))
    # Bigram for Chinese characters to capture multi-character terms
    bigrams: list[str] = []
    cn_chars = [t for t in tokens if not t.isascii()]
    for i in range(len(cn_chars) - 1):
        bigrams.append(cn_chars[i] + cn_chars[i + 1])
    return tokens + bigrams


def build_bm25_index(chunks_jsonl_path: str) -> BM25Retriever:
    from embedding_indexing.io import iter_chunks
    from pathlib import Path

    corpus: list[dict] = []
    for chunk in iter_chunks(Path(chunks_jsonl_path)):
        if chunk.chunk_text and len(chunk.chunk_text.strip()) > 20:
            corpus.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.chunk_text,
                "title": chunk.title,
                "url": chunk.url,
                "section_path": chunk.section_path,
            })

    bm25 = BM25Retriever()
    bm25.index_chunks(corpus)
    return bm25
