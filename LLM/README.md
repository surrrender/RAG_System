# LLM RAG QA

基于本地 `Ollama` 和现有 `Embedding_Indexing` Qdrant 索引的单轮问答模块。

默认会复用 `Embedding_Indexing` 的检索链路：

- embedding 模型：`BAAI/bge-m3`
- reranker 模型：`BAAI/bge-reranker-base`
- 检索流程：`dense 召回 -> rerank 重排 -> 送入 LLM 生成`

## 安装

推荐先在仓库根目录创建并激活统一 Python 环境，详见 [根目录 README](/Users/shizhuo/Documents/Study/RAG/RAG_System/README.md)：

```bash
python3 scripts/bootstrap_python_workspace.py
source .venv/bin/activate
```

如果是首次运行，并且本地还没有缓存 embedding / reranker 模型，请先保证 `Embedding_Indexing` 侧模型已经下载完成。

## CLI

```bash
python -m llm ask "小程序 App 生命周期是什么？"
```

启动 HTTP API：

```bash
python -m llm serve
```

默认接口：

```bash
POST /qa/stream
```
