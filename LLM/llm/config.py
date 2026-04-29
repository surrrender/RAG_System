from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_QDRANT_PATH = (REPO_ROOT / "Embedding_Indexing" / "data" / "qdrant").resolve()
DEFAULT_SQLITE_PATH = (REPO_ROOT / "LLM" / "data" / "app.sqlite3").resolve()
DEFAULT_COLLECTION_NAME = "wechat_framework_chunks"
DEFAULT_EMBEDDER_PROVIDER = "sentence-transformer"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_RERANKER_PROVIDER = "cross-encoder"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANK_CANDIDATE_LIMIT = 5
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 5000
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
# DEFAULT_GENERATION_MODEL = "qwen2.5:7b"
# DEFAULT_GENERATION_MODEL = "llama3.1:8b"
DEFAULT_GENERATION_MODEL = "qwen3:8b"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_REQUEST_TIMEOUT = 60.0


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    qdrant_path: Path = Path(os.getenv("LLM_QDRANT_PATH", DEFAULT_QDRANT_PATH))
    qdrant_url: str | None = os.getenv("LLM_QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("LLM_QDRANT_API_KEY")
    collection_name: str = os.getenv("LLM_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
    embedder_provider: str = os.getenv("LLM_EMBEDDER_PROVIDER", DEFAULT_EMBEDDER_PROVIDER)
    embedding_model: str = os.getenv("LLM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_device: str = os.getenv("LLM_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    reranker_provider: str = os.getenv("LLM_RERANKER_PROVIDER", DEFAULT_RERANKER_PROVIDER)
    reranker_model: str = os.getenv("LLM_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    reranker_device: str = os.getenv("LLM_RERANKER_DEVICE", DEFAULT_RERANKER_DEVICE)
    rerank_candidate_limit: int = _env_int("LLM_RERANK_CANDIDATE_LIMIT", DEFAULT_RERANK_CANDIDATE_LIMIT)
    disable_reranker: bool = _env_bool("LLM_DISABLE_RERANKER", False)
    top_k: int = _env_int("LLM_TOP_K", DEFAULT_TOP_K)
    max_context_chars: int = _env_int("LLM_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS)
    ollama_host: str = os.getenv("LLM_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    generation_model: str = os.getenv("LLM_GENERATION_MODEL", DEFAULT_GENERATION_MODEL)
    http_host: str = os.getenv("LLM_HTTP_HOST", DEFAULT_HTTP_HOST)
    http_port: int = _env_int("LLM_HTTP_PORT", DEFAULT_HTTP_PORT)
    request_timeout: float = _env_float("LLM_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
    sqlite_path: Path = Path(os.getenv("LLM_SQLITE_PATH", DEFAULT_SQLITE_PATH))


def load_settings() -> Settings:
    return Settings()
