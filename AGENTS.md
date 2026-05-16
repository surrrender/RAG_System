# AGENTS.md — RAG_System

## Project structure

Monorepo with 4 subprojects:

| Directory | Language | Entrypoint |
|-----------|----------|------------|
| `Crawler/` | Python | `crawler.cli:app` → CLI: `crawler` |
| `Embedding_Indexing/` | Python | `embedding_indexing.cli:app` → CLI: `embedding-indexing` |
| `LLM/` | Python | `llm.cli:app` → CLI: `llm`; also `python -m llm` |
| `FrontEnd/` | TypeScript (React 19 + Vite) | `index.html` → `src/main.tsx` |

## Data flow

`Crawler` scrapes WeChat dev docs → JSONL → `Embedding_Indexing` vectorizes & stores in local Qdrant → `LLM` retrieves + generates via Ollama.

## Python workspace

Unified venv at root `.venv/` (not per-subproject). Bootstrap:

```bash
python3 scripts/bootstrap_python_workspace.py
source .venv/bin/activate
```

Three packages installed as editable via `requirements-workspace.txt` (`-e ./Crawler`, `-e ./Embedding_Indexing`, `-e ./LLM`). Always activate the root `.venv` before working on any Python subproject.

A constraints file with all pinned transitive deps lives at `.workspace/constraints-from-subvenvs.txt`.

## Key commands

### Python (after `source .venv/bin/activate`)

| Command | What |
|---------|------|
| `crawler --help` | Crawler CLI |
| `embedding-indexing index <path>` | Index JSONL chunks into Qdrant |
| `embedding-indexing search <query>` | Search Qdrant index |
| `python -m llm ask <question>` | Ask a question via Ollama RAG |
| `python -m llm serve` | Start FastAPI server (default `127.0.0.1:8000`) |
| `python -m pytest <path>` | Run Python tests for any package |

**Startup order** when running full stack: `llm serve` first (backend), then FrontEnd dev server.

### FrontEnd

```bash
cd FrontEnd
npm run dev          # Vite dev server, proxies /api → http://127.0.0.1:8000
npm run build        # tsc -b && vite build
npm run typecheck    # tsc --noEmit
npm run test         # vitest run
npm run test:e2e     # Playwright E2E
```

### LLM backend during FrontEnd dev

Start backend first: `python -m llm serve` (listens on `127.0.0.1:8000`). Vite proxies `/api/*` there.

### Benchmarks (FrontEnd)

```bash
npm run benchmark:latency   # Playwright latency test
npm run benchmark:file      # Script-based question benchmark
```

## Configuration & quirks

- **Qdrant**: local mode by default, data in `Embedding_Indexing/embedding_indexing/data/`
- **Embedding model**: `BAAI/bge-small-zh-v1.5` (was `bge-m3`, fallback exists in code)
- **Reranker**: `BAAI/bge-reranker-base`, candidate count = 5
- **Ollama model**: `qwen3:8b` (set via env `LLM_OLLAMA_MODEL`)
- **LLM settings** env vars: `LLM_QDRANT_PATH`, `LLM_OLLAMA_HOST`, `LLM_OLLAMA_MODEL` (see `LLM/llm/config.py`)
- **LLM warmup**: backend pre-warms embedder/reranker on startup
- **SQLite storage**: WAL mode, concurrent-friendly; data in `LLM/llm/data/app.sqlite3`
- **No linter/formatter configs** exist (no ruff, flake8, eslint, prettier — tooling is minimal)
- **No CI** — `.github/workflows/` does not exist

## Testing

- Python: `pytest` (all packages), no `coverage` config, no runner script
- FrontEnd: `vitest` for unit, `playwright` for E2E/benchmarks
- LLM concurrency benchmark: `python LLM/scripts/run_stream_concurrency_benchmark.py`

## Run order for full-stack dev

1. `source .venv/bin/activate`
2. `python -m llm serve` (backend API)
3. In another terminal: `cd FrontEnd && npm run dev` (Vite, opens browser)
