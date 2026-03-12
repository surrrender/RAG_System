from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embedding_indexing.embeddings import BaseEmbedder, build_embedder, chunk_to_embedding_text
from embedding_indexing.io import load_chunks
from embedding_indexing.qdrant_store import QdrantChunkIndex


@dataclass(slots=True)
class IndexStats:
    chunk_count: int
    vector_size: int
    collection_name: str
    qdrant_path: Path


def index_chunks(
    input_path: Path,
    qdrant_path: Path,
    collection_name: str,
    embedder: BaseEmbedder,
    batch_size: int = 32,
    recreate: bool = True,
) -> IndexStats:
    chunks = load_chunks(input_path)
    texts = [chunk_to_embedding_text(chunk) for chunk in chunks]
    # 将文本向量化的地方
    vectors = embedder.embed_texts(texts)

    index = QdrantChunkIndex(path=qdrant_path, collection_name=collection_name)
    index.ensure_collection(vector_size=embedder.dimension, recreate=recreate)
    index.upsert(chunks=chunks, vectors=vectors, batch_size=batch_size)

    return IndexStats(
        chunk_count=len(chunks),
        vector_size=embedder.dimension,
        collection_name=collection_name,
        qdrant_path=qdrant_path,
    )


def search_chunks(
    qdrant_path: Path,
    collection_name: str,
    embedder: BaseEmbedder,
    query: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    index = QdrantChunkIndex(path=qdrant_path, collection_name=collection_name)
    index.ensure_collection(vector_size=embedder.dimension, recreate=False)
    query_vector = embedder.embed_query(query)
    points = index.search(query_vector=query_vector, limit=limit)
    return [
        {
            "score": float(point.score),
            "chunk_id": str(point.payload.get("chunk_id") or point.id),
            "title": point.payload.get("title"),
            "url": point.payload.get("url"),
            "section_path": point.payload.get("section_path"),
            "text": point.payload.get("text"),
        }
        for point in points
    ]


def build_default_embedder(
    provider: str,
    model_name: str,
    hash_dimension: int = 32,
    offline: bool = False,
) -> BaseEmbedder:
    return build_embedder(
        provider=provider,
        model_name=model_name,
        hash_dimension=hash_dimension,
        offline=offline,
    )
