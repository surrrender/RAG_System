import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embedding_indexing.qdrant_store import QdrantChunkIndex


def test_qdrant_local_mode_concurrency_error_mentions_server_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQdrantClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs == {"path": str(Path("/tmp/qdrant"))}
            raise RuntimeError("Storage folder qdrant/ is already accessed by another instance of Qdrant client.")

    fake_models = SimpleNamespace()
    fake_module = ModuleType("qdrant_client")
    fake_module.QdrantClient = FakeQdrantClient
    fake_http_module = ModuleType("qdrant_client.http")
    fake_http_module.models = fake_models
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", fake_http_module)

    with pytest.raises(RuntimeError) as exc_info:
        QdrantChunkIndex(path=Path("/tmp/qdrant"), collection_name="demo")

    message = str(exc_info.value)
    assert "LLM_QDRANT_URL" in message
    assert "local mode" in message
