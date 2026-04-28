from __future__ import annotations

import json
from pathlib import Path

import typer

from embedding_indexing.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INPUT_PATH,
    DEFAULT_LIMIT,
    DEFAULT_MODEL_NAME,
    DEFAULT_QDRANT_API_KEY,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QDRANT_URL,
    DEFAULT_RERANK_CANDIDATE_LIMIT,
    DEFAULT_RERANKER_MODEL_NAME,
)
from embedding_indexing.pipeline import (
    build_default_embedder,
    build_default_reranker,
    initialize_chunk_index,
    index_chunks,
    search_chunks,
)

app = typer.Typer(help="Embedding and local Qdrant indexing for chunked docs.")


@app.command()
def index(
    input_path: Path = typer.Option(DEFAULT_INPUT_PATH, exists=True, file_okay=True, dir_okay=False),
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT_PATH),
    qdrant_url: str | None = typer.Option(DEFAULT_QDRANT_URL),
    qdrant_api_key: str | None = typer.Option(DEFAULT_QDRANT_API_KEY),
    collection_name: str = typer.Option(DEFAULT_COLLECTION_NAME),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME),
    embedder_provider: str = typer.Option("sentence-transformer"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, min=1),
    recreate: bool = typer.Option(False, help="Delete and recreate the target collection before upsert."),
    hash_dimension: int = typer.Option(64, min=4, help="Only used by the hash embedder."),
    offline: bool = typer.Option(False, help="Load embedding model from local cache only."),
    device: str = typer.Option("cpu", help="Embedding device: cpu, mps, cuda, or auto."),
) -> None:
    if not qdrant_url:
        qdrant_path.mkdir(parents=True, exist_ok=True)
    embedder = build_default_embedder(
        provider=embedder_provider,
        model_name=model_name,
        hash_dimension=hash_dimension,
        offline=offline,
        device=device,
    )
    stats = index_chunks(
        input_path=input_path,
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        embedder=embedder,
        batch_size=batch_size,
        recreate=recreate,
    )
    typer.echo(
        f"indexed {stats.chunk_count} chunks into {stats.collection_name} "
        f"(dim={stats.vector_size}) at {stats.qdrant_path}"
    )


# search时默认 --offline,只从本地缓存加载模型，而不主动访问HF hub
@app.command()
def search(
    query: str,
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT_PATH),
    qdrant_url: str | None = typer.Option(DEFAULT_QDRANT_URL),
    qdrant_api_key: str | None = typer.Option(DEFAULT_QDRANT_API_KEY),
    collection_name: str = typer.Option(DEFAULT_COLLECTION_NAME),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME),
    embedder_provider: str = typer.Option("sentence-transformer"),
    limit: int = typer.Option(DEFAULT_LIMIT, min=1),
    hash_dimension: int = typer.Option(64, min=4, help="Only used by the hash embedder."),
    offline: bool = typer.Option(True, help="Load embedding model from local cache only."),
    device: str = typer.Option("cpu", help="Embedding device: cpu, mps, cuda, or auto."),
    reranker_provider: str = typer.Option("cross-encoder"),
    reranker_model_name: str = typer.Option(DEFAULT_RERANKER_MODEL_NAME),
    rerank_candidate_limit: int = typer.Option(DEFAULT_RERANK_CANDIDATE_LIMIT, min=1),
    reranker_offline: bool = typer.Option(True, help="Load reranker model from local cache only."),
    reranker_device: str = typer.Option("cpu", help="Reranker device: cpu, mps, cuda, or auto."),
    disable_reranker: bool = typer.Option(False, help="Skip reranking and return dense retrieval results."),
) -> None:
    if not qdrant_url and not qdrant_path.exists():
        raise typer.BadParameter("qdrant_path does not exist when qdrant_url is not configured.")
    embedder = build_default_embedder(
        provider=embedder_provider,
        model_name=model_name,
        hash_dimension=hash_dimension,
        offline=offline,
        device=device,
    )
    reranker = None
    if not disable_reranker:
        reranker = build_default_reranker(
            provider=reranker_provider,
            model_name=reranker_model_name,
            offline=reranker_offline,
            device=reranker_device,
        )
    index = initialize_chunk_index(
        path=qdrant_path,
        collection_name=collection_name,
        vector_size=embedder.dimension,
        url=qdrant_url,
        api_key=qdrant_api_key,
        recreate=False,
    )
    results = search_chunks(
        index=index,
        embedder=embedder,
        query=query,
        limit=limit,
        reranker=reranker,
        enable_reranker=not disable_reranker,
        rerank_candidate_limit=rerank_candidate_limit,
    )
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
