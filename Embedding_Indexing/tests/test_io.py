from __future__ import annotations

from pathlib import Path

from embedding_indexing.io import load_chunks


def test_load_chunks_reads_jsonl() -> None:
    sample = Path("tests/fixtures/sample_chunks.jsonl")
    chunks = load_chunks(sample)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].section_path == ["注册"]
    assert chunks[0].chunk_type == "text"
