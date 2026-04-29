from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from embedding_indexing import cli


runner = CliRunner()


def test_search_cli_enables_reranker_by_default(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeEmbedder:
        dimension = 16

    def fake_embedder(**kwargs):
        calls["embedder"] = kwargs
        return FakeEmbedder()

    def fake_reranker(**kwargs):
        calls["reranker"] = kwargs
        return object()

    def fake_search_chunks(**kwargs):
        calls["search"] = kwargs
        return [{"score": 1.0, "chunk_id": "c1"}]

    def fake_initialize_chunk_index(**kwargs):
        calls["index"] = kwargs
        return "INDEX"

    monkeypatch.setattr(cli, "build_default_embedder", fake_embedder)
    monkeypatch.setattr(cli, "build_default_reranker", fake_reranker)
    monkeypatch.setattr(cli, "initialize_chunk_index", fake_initialize_chunk_index)
    monkeypatch.setattr(cli, "search_chunks", fake_search_chunks)

    result = runner.invoke(
        cli.app,
        [
            "search",
            "app lifecycle",
            "--qdrant-path",
            str(Path.cwd()),
        ],
    )

    assert result.exit_code == 0
    assert calls["reranker"] == {
        "provider": "cross-encoder",
        "model_name": "BAAI/bge-reranker-base",
        "offline": True,
        "device": "cpu",
    }
    assert calls["embedder"]["device"] == "cpu"
    assert calls["index"]["vector_size"] == 16
    assert calls["search"]["index"] == "INDEX"
    assert calls["search"]["enable_reranker"] is True
    assert calls["search"]["rerank_candidate_limit"] == 10


def test_search_cli_can_disable_reranker(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeEmbedder:
        dimension = 12

    def fake_embedder(**kwargs):
        return FakeEmbedder()

    def fake_search_chunks(**kwargs):
        calls["search"] = kwargs
        return [{"score": 1.0, "chunk_id": "c1"}]

    def fake_initialize_chunk_index(**kwargs):
        calls["index"] = kwargs
        return "INDEX"

    monkeypatch.setattr(cli, "build_default_embedder", fake_embedder)
    monkeypatch.setattr(cli, "initialize_chunk_index", fake_initialize_chunk_index)
    monkeypatch.setattr(cli, "search_chunks", fake_search_chunks)

    def fail_build_reranker(**kwargs):
        raise AssertionError("reranker should not be built when disabled")

    monkeypatch.setattr(cli, "build_default_reranker", fail_build_reranker)

    result = runner.invoke(
        cli.app,
        [
            "search",
            "app lifecycle",
            "--qdrant-path",
            str(Path.cwd()),
            "--disable-reranker",
        ],
    )

    assert result.exit_code == 0
    assert calls["index"]["vector_size"] == 12
    assert calls["search"]["index"] == "INDEX"
    assert calls["search"]["enable_reranker"] is False
    assert calls["search"]["reranker"] is None


def test_index_cli_uses_cpu_device_by_default(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_embedder(**kwargs):
        calls["embedder"] = kwargs
        return object()

    def fake_index_chunks(**kwargs):
        calls["index"] = kwargs
        return type(
            "Stats",
            (),
            {
                "chunk_count": 1,
                "collection_name": "chunks",
                "vector_size": 8,
                "qdrant_path": Path.cwd(),
            },
        )()

    input_path = Path(__file__)
    monkeypatch.setattr(cli, "build_default_embedder", fake_embedder)
    monkeypatch.setattr(cli, "index_chunks", fake_index_chunks)

    result = runner.invoke(
        cli.app,
        [
            "index",
            "--input-path",
            str(input_path),
            "--qdrant-path",
            str(Path.cwd()),
        ],
    )

    assert result.exit_code == 0
    assert calls["embedder"]["device"] == "cpu"
