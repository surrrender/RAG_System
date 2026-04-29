from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from embedding_indexing.models import ChunkRecord


LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}


def chunk_to_payload(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "text": chunk.chunk_text,
        "doc_id": chunk.doc_id,
        "url": chunk.url,
        "title": chunk.title,
        "nav_path": chunk.nav_path,
        "section_path": chunk.section_path,
        "related_code_ids": chunk.related_code_ids,
        "related_text_ids": chunk.related_text_ids,
        "token_estimate": chunk.token_estimate,
        "fetched_at": chunk.fetched_at,
    }


def chunk_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantChunkIndex:
    def __init__(
        self,
        path: Path,
        collection_name: str,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
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
        self._validated_vector_size: int | None = None
        self._configured_path = path
        self._configured_url = _normalize_local_service_url(url) or url
        if url:
            try:
                self.client = QdrantClient(url=self._configured_url, api_key=api_key)
            except Exception as exc:
                raise RuntimeError(self._build_connection_error(exc)) from exc
        else:
            try:
                self.client = QdrantClient(path=str(path))
            except Exception as exc:
                if _is_local_mode_concurrency_error(exc):
                    raise RuntimeError(self._build_local_mode_concurrency_error(exc)) from exc
                raise RuntimeError(self._build_connection_error(exc)) from exc

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as exc:
            raise RuntimeError(self._build_connection_error(exc)) from exc
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._rest.VectorParams(size=vector_size, distance=self._rest.Distance.COSINE),
            )
            self._validated_vector_size = int(vector_size)
            return

        try:
            collection_info = self.client.get_collection(self.collection_name)
        except Exception as exc:
            raise RuntimeError(self._build_connection_error(exc)) from exc
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
        self._validated_vector_size = int(vector_size)

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
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
            except Exception as exc:
                raise RuntimeError(self._build_connection_error(exc)) from exc

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        chunk_type: str | None = None,
    ) -> list[object]:
        if self._validated_vector_size is not None and len(query_vector) != self._validated_vector_size:
            raise RuntimeError(
                f"Collection '{self.collection_name}' expects query vectors of size "
                f"{self._validated_vector_size}, but received {len(query_vector)}."
            )
        query_filter = None
        if chunk_type is not None:
            query_filter = self._rest.Filter(
                must=[
                    self._rest.FieldCondition(
                        key="chunk_type",
                        match=self._rest.MatchValue(value=chunk_type),
                    )
                ]
            )

        try:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            ).points
        except Exception as exc:
            raise RuntimeError(self._build_connection_error(exc)) from exc

    def _build_connection_error(self, exc: Exception) -> str:
        return f"Failed to reach Qdrant: {exc}. {_protocol_hint(self._configured_url, 'Qdrant')}"

    def _build_local_mode_concurrency_error(self, exc: Exception) -> str:
        return (
            f"Failed to open local Qdrant path '{self._configured_path}': {exc}. "
            "Qdrant local mode does not support concurrent multi-client access; "
            "for multi-user QA please run Qdrant server and set LLM_QDRANT_URL."
        )


def _normalize_local_service_url(url: str | None) -> str | None:
    if url is None:
        return None

    trimmed = url.strip()
    if not trimmed:
        return None

    parsed = urlsplit(trimmed)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and hostname in LOCAL_HOSTS:
        return urlunsplit(("http", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return trimmed


def _protocol_hint(url: str | None, service_name: str) -> str:
    normalized = _normalize_local_service_url(url)
    if not normalized:
        return f"{service_name} connection failed."

    parsed = urlsplit(normalized)
    host = parsed.hostname or normalized
    if host.lower() in LOCAL_HOSTS:
        return (
            f"{service_name} connection failed. Check whether the service is running and whether "
            f"the URL scheme should be http:// instead of https://."
        )
    return f"{service_name} connection failed. Check whether the configured URL is reachable."


def _is_local_mode_concurrency_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "already accessed by another instance" in message or ".lock" in message
