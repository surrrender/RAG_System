from __future__ import annotations

import uuid
from pathlib import Path

from embedding_indexing.models import ChunkRecord


def chunk_to_payload(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.chunk_text,
        "doc_id": chunk.doc_id,
        "url": chunk.url,
        "title": chunk.title,
        "nav_path": chunk.nav_path,
        "section_path": chunk.section_path,
        "code_blocks": chunk.code_blocks,
        "token_estimate": chunk.token_estimate,
        "fetched_at": chunk.fetched_at,
    }


def chunk_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantChunkIndex:
    def __init__(self, path: Path, collection_name: str) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for indexing and search. "
                "Install project dependencies before using the Qdrant store."
            ) from exc

        self.collection_name = collection_name
        self._rest = rest
        self.client = QdrantClient(path=str(path))

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._rest.VectorParams(size=vector_size, distance=self._rest.Distance.COSINE),
            )
            return

        collection_info = self.client.get_collection(self.collection_name)
        configured_vectors = collection_info.config.params.vectors
        current_size = getattr(configured_vectors, "size", None)
        if current_size is None:
            raise RuntimeError(
                f"Collection '{self.collection_name}' has an unsupported vector configuration."
            )
        if int(current_size) != int(vector_size):
            raise RuntimeError(
                f"Collection '{self.collection_name}' already exists with vector size {current_size}, "
                f"but the current embedder produces size {vector_size}. "
                f"Use --recreate to rebuild the collection, or choose a different --qdrant-path / "
                f"--collection-name."
            )

    def upsert(self, chunks: list[ChunkRecord], vectors: list[list[float]], batch_size: int = 64) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match.")

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            batch_chunks = chunks[start:end]
            batch_vectors = vectors[start:end]
            points = [
                self._rest.PointStruct(id=chunk_to_point_id(chunk.chunk_id), vector=vector, payload=chunk_to_payload(chunk))
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit: int = 5) -> list[object]:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points
