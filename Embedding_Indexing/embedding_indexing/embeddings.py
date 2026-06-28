from __future__ import annotations

import os
from abc import ABC, abstractmethod

from embedding_indexing.models import ChunkRecord


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, offline: bool = False, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the default embedder. "
                "Install project dependencies before indexing."
            ) from exc

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self._model = SentenceTransformer(
            model_name,
            local_files_only=offline,
            device=device,
        )
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


def chunk_to_embedding_text(chunk: ChunkRecord) -> str:
    parts = [
        chunk.title.strip(),
        _format_chunk_path(chunk.nav_path, chunk.section_path),
        chunk.chunk_text.strip(),
    ]
    return "\n".join(part for part in parts if part).strip()


def _format_chunk_path(nav_path: list[str], section_path: list[str]) -> str:
    full_path = [item.strip() for item in [*nav_path, *section_path] if str(item).strip()]
    return " > ".join(full_path)


def build_embedder(
    provider: str,
    model_name: str,
    offline: bool = False,
    device: str = "cpu",
) -> BaseEmbedder:
    if provider == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name, offline=offline, device=device)
    raise ValueError(f"Unsupported embedder provider: {provider}")
