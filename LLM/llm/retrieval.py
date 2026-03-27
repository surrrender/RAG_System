from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm._embedding_indexing import load_embedding_indexing_symbols
from llm.models import RetrievedChunk


@dataclass(slots=True)
class Retriever:
    qdrant_path: Path
    collection_name: str
    embedder_provider: str
    embedding_model: str
    reranker_provider: str
    reranker_model: str
    rerank_candidate_limit: int = 10
    disable_reranker: bool = False

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        build_default_embedder, build_default_reranker, search_chunks = load_embedding_indexing_symbols()
        embedder = build_default_embedder(
            provider=self.embedder_provider,
            model_name=self.embedding_model,
            offline=True,
        )
        reranker = None
        if not self.disable_reranker:
            reranker = build_default_reranker(
                provider=self.reranker_provider,
                model_name=self.reranker_model,
                offline=True,
            )
        results = search_chunks(
            qdrant_path=self.qdrant_path,
            collection_name=self.collection_name,
            embedder=embedder,
            query=question,
            limit=top_k,
            reranker=reranker,
            enable_reranker=not self.disable_reranker,
            rerank_candidate_limit=self.rerank_candidate_limit,
        )
        chunks = [
            RetrievedChunk(
                chunk_id=str(item["chunk_id"]),
                score=float(item["score"]),
                title=_optional_str(item.get("title")),
                url=_optional_str(item.get("url")),
                section_path=_optional_list(item.get("section_path")),
                text=_optional_str(item.get("chunk_text")),
            )
            for item in results
        ]
        return sorted(chunks, key=lambda item: item.score, reverse=True)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_list(value: object) -> list[str] | None:
    if value is None:
        return None
    items = [str(item) for item in value if str(item).strip()]
    return items or None
