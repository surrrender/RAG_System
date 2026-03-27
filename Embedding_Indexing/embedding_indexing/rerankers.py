from __future__ import annotations

import os
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str, offline: bool = False) -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the cross-encoder reranker. "
                "Install project dependencies before using reranking."
            ) from exc

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

        self._model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            local_files_only=offline,
        )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        pairs = [[query, document] for document in documents]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


def build_reranker(provider: str, model_name: str, offline: bool = False) -> BaseReranker:
    if provider == "cross-encoder":
        return CrossEncoderReranker(model_name=model_name, offline=offline)
    raise ValueError(f"Unsupported reranker provider: {provider}")
