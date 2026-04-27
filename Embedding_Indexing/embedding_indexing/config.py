from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = (PROJECT_ROOT.parent / "Crawler" / "outputs" / "framework_chunks.jsonl").resolve()
DEFAULT_QDRANT_PATH = (PROJECT_ROOT / "data" / "qdrant").resolve()
DEFAULT_COLLECTION_NAME = "wechat_framework_chunks"
# DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_BATCH_SIZE = 4
DEFAULT_LIMIT = 5
DEFAULT_RERANK_CANDIDATE_LIMIT = 10
DEFAULT_QDRANT_URL = os.getenv("EMBEDDING_QDRANT_URL")
DEFAULT_QDRANT_API_KEY = os.getenv("EMBEDDING_QDRANT_API_KEY")
