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

## 多用户与多会话

当前 HTTP API 已支持轻量 `user_id` + `conversation_id` 的多用户多会话模式。

主要接口：

```bash
GET    /conversations?user_id=...
POST   /conversations
PATCH  /conversations/{conversation_id}
DELETE /conversations/{conversation_id}?user_id=...
GET    /conversations/{conversation_id}/messages?user_id=...
POST   /qa
POST   /qa/stream
```

`/qa` 与 `/qa/stream` 请求体现在都需要：

```json
{
  "user_id": "browser-local-user-id",
  "conversation_id": "uuid",
  "question": "小程序 App 生命周期是什么？",
  "top_k": 5
}
```

会话与消息历史默认存储在本地 SQLite：

```bash
LLM_SQLITE_PATH=/absolute/path/to/app.sqlite3
```

当前 SQLite 连接会显式启用：

- `journal_mode=WAL`
- `synchronous=NORMAL`
- `busy_timeout=5000`
- 写事务使用 `BEGIN IMMEDIATE`

这样可以降低多用户并发时的写锁冲突，并避免读后写事务在竞争下更容易出现的锁升级失败。

## Qdrant 配置

默认仍然兼容本地目录模式：

```bash
LLM_QDRANT_PATH=./Embedding_Indexing/data/qdrant
```

如果要切到多人并发更适合的 Qdrant server 模式，可以改为：

```bash
LLM_QDRANT_URL=http://127.0.0.1:6333
LLM_QDRANT_API_KEY=optional-api-key
```

补充说明：

- `Qdrant local mode` 更适合开发和单实例测试。
- 如果同一个本地索引目录会被多个客户端、多个进程或更高并发同时访问，建议改用 `Qdrant server mode`。
- 现在如果检测到 local mode 目录锁冲突，错误信息会直接提示切换到 `LLM_QDRANT_URL`。

## 并发压测

仓库里新增了一个可直接执行的并发压测脚本：

```bash
python LLM/scripts/run_stream_concurrency_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --user-count 4 \
  --conversations-per-user 2 \
  --question "小程序 App 生命周期是什么？"
```

它会先自动创建多组 `user_id` / `conversation_id`，然后并发发起 `/qa/stream` 请求，并输出：

- 失败率
- 首字符延迟统计
- 单请求总耗时统计
- 服务端检索/首 token/总耗时统计
