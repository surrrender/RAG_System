from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from embedding_indexing.embeddings import BaseEmbedder, build_embedder, chunk_to_embedding_text
from embedding_indexing.io import iter_chunks
from embedding_indexing.models import ChunkRecord
from embedding_indexing.qdrant_store import QdrantChunkIndex
from embedding_indexing.rerankers import BaseReranker, build_reranker


@dataclass(slots=True)
class IndexStats:
    chunk_count: int
    vector_size: int
    collection_name: str
    qdrant_path: Path


def index_chunks(
    input_path: Path,
    qdrant_path: Path,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    collection_name: str,
    embedder: BaseEmbedder,
    batch_size: int = 32,
    recreate: bool = True,
) -> IndexStats:
    index = initialize_chunk_index(
        path=qdrant_path,
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
        vector_size=embedder.dimension,
        recreate=recreate,
    )
    chunk_count = 0

    for chunk_batch in _batched_chunks(iter_chunks(input_path), batch_size=batch_size):
        texts = [chunk_to_embedding_text(chunk) for chunk in chunk_batch]
        vectors = embedder.embed_texts(texts)
        index.upsert(chunks=chunk_batch, vectors=vectors, batch_size=len(chunk_batch))
        chunk_count += len(chunk_batch)

    return IndexStats(
        chunk_count=chunk_count,
        vector_size=embedder.dimension,
        collection_name=collection_name,
        qdrant_path=qdrant_path,
    )


def build_chunk_index(
    qdrant_path: Path,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    collection_name: str,
) -> QdrantChunkIndex:
    return QdrantChunkIndex(
        path=qdrant_path,
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
    )


def initialize_chunk_index(
    path: Path,
    collection_name: str,
    vector_size: int,
    url: str | None = None,
    api_key: str | None = None,
    recreate: bool = False,
) -> QdrantChunkIndex:
    # setting up a Qdrant connection plus collection initialization, including vector size validation and optional collection recreation
    index = QdrantChunkIndex(
        path=path,
        collection_name=collection_name,
        url=url,
        api_key=api_key,
    )
    index.ensure_collection(vector_size=vector_size, recreate=recreate)
    return index


def search_chunks(
    index: QdrantChunkIndex,
    embedder: BaseEmbedder,
    query: str,
    limit: int = 5,
    reranker: BaseReranker | None = None,
    enable_reranker: bool = True,
    rerank_candidate_limit: int = 10,
    stage_metrics: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    if enable_reranker and reranker is None:
        raise ValueError("Reranker is enabled, but no reranker instance was provided.")

    embed_started_at = time.perf_counter()
    query_vector = embedder.embed_query(query)
    if stage_metrics is not None:
        stage_metrics["embed_ms"] = _elapsed_ms(embed_started_at)

    candidate_limit = max(limit, rerank_candidate_limit)
    vector_search_started_at = time.perf_counter()
    points = index.search(query_vector=query_vector, limit=candidate_limit)
    if stage_metrics is not None:
        stage_metrics["vector_search_ms"] = _elapsed_ms(vector_search_started_at)

    if not enable_reranker or len(points) <= limit:
        return [_point_to_result(point) for point in points[:limit]]

    rerank_started_at = time.perf_counter()
    ranked_points = _rerank_points(query=query, points=points, reranker=reranker)
    if stage_metrics is not None:
        stage_metrics["rerank_ms"] = _elapsed_ms(rerank_started_at)
    return [_point_to_result(point, score=score) for point, score in ranked_points[:limit]]


def build_default_embedder(
    provider: str,
    model_name: str,
    hash_dimension: int = 32,
    offline: bool = False,
    device: str = "cpu",
) -> BaseEmbedder:
    return build_embedder(
        provider=provider,
        model_name=model_name,
        hash_dimension=hash_dimension,
        offline=offline,
        device=device,
    )


def build_default_reranker(
    provider: str,
    model_name: str,
    offline: bool = False,
    device: str = "cpu",
) -> BaseReranker:
    return build_reranker(
        provider=provider,
        model_name=model_name,
        offline=offline,
        device=device,
    )


def _rerank_points(
    query: str,
    points: list[object],
    reranker: BaseReranker,
) -> list[tuple[object, float]]:
    documents = [str(point.payload.get("text") or "") for point in points]
    scores = reranker.rerank(query=query, documents=documents)
    if len(scores) != len(points):
        raise RuntimeError(
            f"Reranker returned {len(scores)} scores for {len(points)} candidates."
        )

    ranked = list(zip(points, scores, strict=True))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _point_to_result(point: object, score: float | None = None) -> dict[str, object]:
    return {
        "score": float(point.score if score is None else score),
        "chunk_id": str(point.payload.get("chunk_id") or point.id),
        "title": point.payload.get("title"),
        "url": point.payload.get("url"),
        "section_path": point.payload.get("section_path"),
        "chunk_type": point.payload.get("chunk_type"),
        "chunk_text": point.payload.get("text"),
    }


def _elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 3)


# TODO:这里实现了一个批处理生成器,作用是将所有的 chunk 分配进行处理而非一次性导入
def _batched_chunks(chunks: Iterable[ChunkRecord], batch_size: int) -> Iterable[list[ChunkRecord]]:
    batch: list[ChunkRecord] = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch # yield:逐个返回,保存当前状态
            batch = []
    if batch:
        yield batch
