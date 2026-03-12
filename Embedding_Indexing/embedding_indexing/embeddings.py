from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import Iterable

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
    def __init__(self, model_name: str, offline: bool = False) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the default embedder. "
                "Install project dependencies before indexing."
            ) from exc

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

        self._model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            local_files_only=offline,
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


class HashEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        values = [0.0] * self._dimension
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:2], "big") % self._dimension
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            values[slot] += sign

        norm = math.sqrt(sum(item * item for item in values))
        if norm > 0:
            return [item / norm for item in values]
        return values


def _tokenize(text: str) -> Iterable[str]:
    for token in text.split():
        token = token.strip()
        if token:
            yield token


def chunk_to_embedding_text(chunk: ChunkRecord) -> str:
    return chunk.chunk_text.strip()


def build_embedder(provider: str, model_name: str, hash_dimension: int = 32, offline: bool = False) -> BaseEmbedder:
    if provider == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name, offline=offline)
    if provider == "hash":
        return HashEmbedder(dimension=hash_dimension)
    raise ValueError(f"Unsupported embedder provider: {provider}")
