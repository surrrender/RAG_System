# Embedding Indexing

用于将 `Crawler` 产出的微信小程序文档切块数据做向量化，并写入本地 Qdrant 索引，供后续检索与 RAG 使用。

当前实现是单机、本地优先方案：

- 默认 embedding 模型：`BAAI/bge-m3`
- 默认 reranker 模型：`BAAI/bge-reranker-base`
- 默认向量库：`Qdrant local mode`
- 默认输入：`../Crawler/outputs/framework_chunks.jsonl`
- 默认集合名：`wechat_framework_chunks`

## 功能概览

- 读取 `framework_chunks.jsonl`
- 将每条 chunk 的 `chunk_text` 转成 embedding
- 使用稳定 UUID 作为 Qdrant point id，并在 payload 中保留原始 `chunk_id`
- 同一个 `chunk_id` 会映射到同一个 point id，因此重复导入时会稳定 upsert
- 提供本地检索命令，便于快速验证召回结果
- `search` 默认执行 dense 召回后再用 Cross-Encoder reranker 重排
- 提供一个测试用 `hash` embedder，方便在未下载真实模型时做基础联调

## 目录结构

```text
.
├── embedding_indexing/
│   ├── cli.py            # CLI 入口
│   ├── config.py         # 默认路径和参数
│   ├── embeddings.py     # embedding 抽象与实现
│   ├── io.py             # JSONL 读取
│   ├── pipeline.py       # 索引与检索主流程
│   └── qdrant_store.py   # Qdrant local 封装
├── tests/                # 基础测试
└── data/qdrant/          # 默认本地索引目录
```

## 环境要求

- Python `3.11+`
- 推荐使用仓库根目录统一的 `.venv`
- 首次使用默认模型时需要联网下载 Hugging Face 模型文件
- 模型下载完成后，建议查询时设置 `HF_HUB_OFFLINE=1`，避免每次查询都去请求 Hugging Face 元数据

## 安装

推荐先在仓库根目录创建并激活统一 Python 环境，详见 [根目录 README](/Users/shizhuo/Documents/Study/RAG/RAG_System/README.md)：

```bash
python3 scripts/bootstrap_python_workspace.py
source .venv/bin/activate
```

如果你只想单独安装当前子项目，也可以继续执行：

```bash
python -m pip install -e '.[dev]'
```

如果你只想先验证命令和流程，不想立即下载大模型，可以先使用测试 embedder：

```bash
python -m embedding_indexing index --embedder-provider hash --model-name ignored
```

说明：

- `index` 默认允许联网下载模型，因为首次建索引通常需要拉取模型文件
- `search` 默认对 embedding 模型和 reranker 都使用离线模式，只从本地缓存加载模型

## 输入数据格式

默认读取父项目爬虫产物：

```text
../Crawler/outputs/framework_chunks.jsonl
```

每行是一个 chunk 级 JSON，当前索引流程会使用这些字段：

- `chunk_id`
- `doc_id`
- `url`
- `title`
- `nav_path`
- `section_path`
- `chunk_type`
- `chunk_text`
- `related_code_ids`
- `related_text_ids`
- `token_estimate`
- `fetched_at`

其中：

- embedding 默认只对 `chunk_text` 编码，因此文本 chunk 和代码 chunk 都会被单独向量化
- metadata 会完整写入 Qdrant payload
- `chunk_id` 会保留在 Qdrant payload 中
- Qdrant 内部 point id 会由 `chunk_id` 映射为稳定 UUID

## 使用方式

### 1. 建立索引

使用默认配置建立本地索引：

```bash
python -m embedding_indexing index
```

常见参数：

```bash
python -m embedding_indexing index ^
  --input-path ..\Crawler\outputs\framework_chunks.jsonl ^
  --qdrant-path .\data\qdrant ^
  --collection-name wechat_framework_chunks ^
  --model-name BAAI/bge-m3 ^
  --embedder-provider sentence-transformer ^
  --batch-size 32 ^
  --recreate
```

参数说明：

- `--input-path`: chunk JSONL 输入文件
- `--qdrant-path`: 本地 Qdrant 数据目录
- `--collection-name`: 目标 collection 名称
- `--model-name`: embedding 模型名
- `--embedder-provider`: 当前支持 `sentence-transformer` 和 `hash`
- `--batch-size`: 批量写入和编码大小
- `--recreate`: 删除并重建 collection
- `--hash-dimension`: 仅 `hash` embedder 使用
- `--offline`: 仅从本地缓存加载模型；适合模型已下载完成后的离线检索

运行成功后会输出类似：

```text
indexed 456 chunks into wechat_framework_chunks (dim=1024) at D:\...\Embedding_Indexing\data\qdrant
```

注意：

- 如果你之前用 `hash` embedder 或旧的 embedding 模型建过索引，再切换到 `BAAI/bge-m3` 时，必须使用 `--recreate`
- 或者改用新的 `--qdrant-path` / `--collection-name`
- 原因是旧集合的向量维度很可能与当前默认模型不同，不能混用

### 2. 查询检索结果

```bash
python -m embedding_indexing search "小程序 App 生命周期"
```

也可以显式指定参数：

```bash
python -m embedding_indexing search "Page onLoad 是什么时候触发的" ^
  --qdrant-path .\data\qdrant ^
  --collection-name wechat_framework_chunks ^
  --model-name BAAI/bge-m3 ^
  --embedder-provider sentence-transformer ^
  --reranker-model-name BAAI/bge-reranker-base ^
  --rerank-candidate-limit 10 ^
  --limit 3
```

说明：

- `search` 命令默认已经开启 `--offline`
- `search` 命令默认也会开启 reranker，并默认开启 `--reranker-offline`
- `--limit` 表示最终返回的结果数量，默认 `5`
- `--rerank-candidate-limit` 表示 dense 阶段先召回多少条候选，再交给 reranker 重排，默认 `10`
- 如果本地还没有缓存 embedding 模型或 reranker 模型，请先在联网状态运行一次查询，或手动传入 `--no-offline --no-reranker-offline`
- 如果只想看原始 dense 结果，可以传入 `--disable-reranker`

输出是 JSON 数组，包含：

- `score`
- `chunk_id`
- `chunk_type`
- `title`
- `url`
- `section_path`
- `chunk_text`

## 默认数据映射

写入 Qdrant 时，字段映射如下：

- `id`: `uuid5(chunk_id)` 生成的稳定 UUID
- `vector`: chunk embedding
- `payload.chunk_id`: `chunk_id`
- `payload.chunk_type`: `chunk_type`
- `payload.text`: `chunk_text`
- `payload.doc_id`: `doc_id`
- `payload.url`: `url`
- `payload.title`: `title`
- `payload.nav_path`: `nav_path`
- `payload.section_path`: `section_path`
- `payload.related_code_ids`: `related_code_ids`
- `payload.related_text_ids`: `related_text_ids`
- `payload.token_estimate`: `token_estimate`
- `payload.fetched_at`: `fetched_at`

这意味着后续如果要做：

- 按 `doc_id` 去重
- 按 `url` 聚合
- 按 `nav_path` 或 `section_path` 过滤

都不需要重做底层数据结构。

## 测试

运行测试：

```bash
python -m pytest -o cache_dir=state/.pytest_cache
```

当前测试包括：

- JSONL chunk 读取
- 基于测试 embedder 的基础索引/检索流程
- search CLI 的 reranker 默认开启与显式关闭行为

说明：

- 如果当前环境没有安装 `qdrant-client`，Qdrant 集成测试会被跳过
- 在当前 Windows 环境下，`pytest` 可能会对默认缓存目录给出权限警告，不影响测试结论

## 已知限制

- 当前默认只编码 `chunk_text`，代码依赖独立的 `code` chunk 参与召回
- 当前没有实现“只重建变更 chunk”的增量索引逻辑
- 首次使用 `BAAI/bge-m3` 或 `BAAI/bge-reranker-base` 会下载模型，耗时取决于网络、磁盘和可用内存
- 如果已有旧的 hash 索引或旧 embedding 模型索引，切换到当前默认模型可能会报维度不匹配；需要 `--recreate` 或换新的索引目录/集合
- `search` 现在会先检查 collection 维度是否和当前模型一致，不一致时会直接给出明确错误
- 检索接口当前直接对整个集合召回并重排，还没有按 `doc_id` 做聚合或去重

## 后续建议

下一步比较合理的增强顺序：

1. 增加增量索引，只处理新增或变化的 `chunk_id`
2. 增加检索评估脚本，固定一批问题做召回验证
3. 增加 metadata filter 和按文档聚合输出
4. 增加更细粒度的检索评估，对 dense 与 rerank 两阶段分别打指标

## 关键文件

- [embedding_indexing/cli.py](/D:/石卓/RAG_demo/Embedding_Indexing/embedding_indexing/cli.py): CLI 入口
- [embedding_indexing/pipeline.py](/D:/石卓/RAG_demo/Embedding_Indexing/embedding_indexing/pipeline.py): 索引与检索流程
- [embedding_indexing/embeddings.py](/D:/石卓/RAG_demo/Embedding_Indexing/embedding_indexing/embeddings.py): embedding 抽象与实现
- [embedding_indexing/qdrant_store.py](/D:/石卓/RAG_demo/Embedding_Indexing/embedding_indexing/qdrant_store.py): Qdrant local 存储封装
- [tests/test_io.py](/D:/石卓/RAG_demo/Embedding_Indexing/tests/test_io.py): JSONL 读取测试
