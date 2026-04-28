from __future__ import annotations

import os
from contextlib import contextmanager
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str, offline: bool = False, device: str = "cpu") -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the cross-encoder reranker. "
                "Install project dependencies before using reranking."
            ) from exc

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

        with _proxy_guard(enabled=offline):
            self._model = CrossEncoder(
                model_name,
                trust_remote_code=True,
                local_files_only=offline,
                device=device,
            )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        pairs = [[query, document] for document in documents]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


def build_reranker(
    provider: str,
    model_name: str,
    offline: bool = False,
    device: str = "cpu",
) -> BaseReranker:
    if provider == "cross-encoder":
        return CrossEncoderReranker(model_name=model_name, offline=offline, device=device)
    raise ValueError(f"Unsupported reranker provider: {provider}")


@contextmanager
def _proxy_guard(enabled: bool):
    if not enabled:
        yield
        return

    proxy_keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    snapshot = {key: os.environ.pop(key, None) for key in proxy_keys}
    try:
        yield
    finally:
        for key, value in snapshot.items():
            if value is not None:
                os.environ[key] = value
