# LLM RAG QA

基于本地 `Ollama` 和现有 `Embedding_Indexing` Qdrant 索引的单轮问答模块。

## 安装

先安装向量检索模块：

```bash
python -m pip install -e ../Embedding_Indexing
```

再安装当前项目：

```bash
python -m pip install -e '.[dev]'
```

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
POST /qa
```
