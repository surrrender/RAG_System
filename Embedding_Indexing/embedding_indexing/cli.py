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
    DEFAULT_QDRANT_PATH,
)
from embedding_indexing.pipeline import build_default_embedder, index_chunks, search_chunks

app = typer.Typer(help="Embedding and local Qdrant indexing for chunked docs.")


@app.command()
def index(
    input_path: Path = typer.Option(DEFAULT_INPUT_PATH, exists=True, file_okay=True, dir_okay=False),
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT_PATH),
    collection_name: str = typer.Option(DEFAULT_COLLECTION_NAME),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME),
    embedder_provider: str = typer.Option("sentence-transformer"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, min=1),
    recreate: bool = typer.Option(False, help="Delete and recreate the target collection before upsert."),
    hash_dimension: int = typer.Option(64, min=4, help="Only used by the hash embedder."),
    offline: bool = typer.Option(False, help="Load embedding model from local cache only."),
) -> None:
    qdrant_path.mkdir(parents=True, exist_ok=True)
    embedder = build_default_embedder(
        provider=embedder_provider,
        model_name=model_name,
        hash_dimension=hash_dimension,
        offline=offline,
    )
    stats = index_chunks(
        input_path=input_path,
        qdrant_path=qdrant_path,
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
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT_PATH, exists=True),
    collection_name: str = typer.Option(DEFAULT_COLLECTION_NAME),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME),
    embedder_provider: str = typer.Option("sentence-transformer"),
    limit: int = typer.Option(DEFAULT_LIMIT, min=1),
    hash_dimension: int = typer.Option(64, min=4, help="Only used by the hash embedder."),
    offline: bool = typer.Option(True, help="Load embedding model from local cache only."),
) -> None:
    embedder = build_default_embedder(
        provider=embedder_provider,
        model_name=model_name,
        hash_dimension=hash_dimension,
        offline=offline,
    )
    results = search_chunks(
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        embedder=embedder,
        query=query,
        limit=limit,
    )
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
