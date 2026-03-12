from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from embedding_indexing.embeddings import HashEmbedder
from embedding_indexing.pipeline import index_chunks, search_chunks


pytest.importorskip("qdrant_client")


def _write_chunks(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"chunk_id":"c1","doc_id":"d1","url":"https://example.com/app","title":"App 生命周期",'
                '"nav_path":["框架"],"section_path":["App"],"chunk_text":"app launch show hide lifecycle",'
                '"code_blocks":[],"token_estimate":5,"fetched_at":"2026-03-12T00:00:00+00:00"}',
                '{"chunk_id":"c2","doc_id":"d2","url":"https://example.com/page","title":"Page 注册",'
                '"nav_path":["框架"],"section_path":["Page"],"chunk_text":"page register onload onshow methods",'
                '"code_blocks":[],"token_estimate":5,"fetched_at":"2026-03-12T00:00:00+00:00"}',
            ]
        ),
        encoding="utf-8",
    )


def test_index_and_search_round_trip() -> None:
    state_root = Path("state/test_tmp")
    state_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=state_root) as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "chunks.jsonl"
        qdrant_path = temp_path / "qdrant"
        _write_chunks(input_path)

        stats = index_chunks(
            input_path=input_path,
            qdrant_path=qdrant_path,
            collection_name="chunks",
            embedder=HashEmbedder(dimension=48),
            recreate=True,
        )

        results = search_chunks(
            qdrant_path=qdrant_path,
            collection_name="chunks",
            embedder=HashEmbedder(dimension=48),
            query="app launch lifecycle",
            limit=1,
        )

    assert stats.chunk_count == 2
    assert results[0]["chunk_id"] == "c1"
