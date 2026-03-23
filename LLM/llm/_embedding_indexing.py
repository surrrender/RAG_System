from __future__ import annotations

import sys
from pathlib import Path


def _add_local_project_path() -> None:
    project_root = Path(__file__).resolve().parents[2] / "Embedding_Indexing"
    project_path = str(project_root)
    if project_root.exists() and project_path not in sys.path:
        sys.path.insert(0, project_path)


def load_embedding_indexing_symbols() -> tuple[object, object]:
    try:
        from embedding_indexing.pipeline import build_default_embedder, search_chunks
    except ModuleNotFoundError:
        _add_local_project_path()
        from embedding_indexing.pipeline import build_default_embedder, search_chunks

    return build_default_embedder, search_chunks
