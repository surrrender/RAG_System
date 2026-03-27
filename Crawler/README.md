# WeChat Mini Program Framework Crawler

用于抓取微信小程序文档 `reference` 下侧边栏“框架”目录的页面内容，并产出适合 RAG 入库的页面级和切块级 JSONL 数据。

## 功能概览

- 基于 Playwright 打开 `https://developers.weixin.qq.com/miniprogram/dev/reference/`
- 从侧边栏定位“框架”目录并提取文档链接
- 逐页抓取正文、标题、导航路径、更新时间、代码块
- 按 `h2/h3/h4` 标题层级切块
- 文本内容与代码示例分开产出，代码也会单独成为可索引 chunk
- 输出页面级、切块级、失败记录三类 JSONL
- 支持 `full` 全量重爬和 `incremental` 增量输出

## 目录结构

```text
.
├─ crawler/    # 核心实现与测试
├─ outputs/    # 爬取产物
└─ state/      # 增量状态
```

## 环境要求

- Python `3.11+`
- Windows / Linux / macOS 均可，当前已在 Windows PowerShell 环境下实现和验证
- 需要安装 Playwright Chromium 浏览器

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

安装 Playwright Chromium：

```bash
python -m playwright install chromium
```

## 运行方式

全量模式：

```bash
python -m crawler --mode full
```

增量模式：

```bash
python -m crawler run --mode incremental
```

可选参数：

```bash
python -m crawler run ^
  --mode full ^
  --headless true ^
  --max-concurrency 4 ^
  --timeout-ms 15000 ^
  --include-code true
```

参数说明：

- `--mode`: `full` 或 `incremental`
- `--headless`: 是否无头运行浏览器
- `--max-concurrency`: 页面抓取最大并发数
- `--timeout-ms`: 单页超时时间，单位毫秒
- `--include-code`: 是否保留代码块

运行完成后会输出类似摘要：

```text
crawl complete: discovered=123 fetched=123 chunks=456 failed=0
```

## 输出文件

主产物目录为 `outputs/`。

### `outputs/framework_pages.jsonl`

每行一个页面级 JSON，对应字段：

- `doc_id`
- `url`
- `title`
- `nav_path`
- `raw_text`
- `code_blocks`
- `source`
- `fetched_at`
- `updated_at`

示例：

```json
{
  "doc_id": "1f2a3b4c5d6e7f80",
  "url": "https://developers.weixin.qq.com/miniprogram/dev/reference/api/App.html",
  "title": "App",
  "nav_path": ["文档", "框架", "App"],
  "raw_text": "页面全文内容",
  "code_blocks": ["App({})"],
  "source": "wechat-miniprogram-framework-docs",
  "fetched_at": "2026-03-10T03:00:00+00:00",
  "updated_at": "2026-03-01"
}
```

### `outputs/framework_chunks.jsonl`

每行一个切块级 JSON，对应字段：

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

文本 chunk 示例：

```json
{
  "chunk_id": "1f2a3b4c5d6e7f80-aabbccddeeff00112233",
  "doc_id": "1f2a3b4c5d6e7f80",
  "url": "https://developers.weixin.qq.com/miniprogram/dev/reference/api/App.html",
  "title": "App",
  "nav_path": ["文档", "框架", "App"],
  "section_path": ["注册", "参数"],
  "chunk_type": "text",
  "chunk_text": "该章节正文",
  "related_code_ids": ["1f2a3b4c5d6e7f80-ffeeddccbbaa00998877"],
  "related_text_ids": [],
  "token_estimate": 42,
  "fetched_at": "2026-03-10T03:00:00+00:00"
}
```

代码 chunk 示例：

```json
{
  "chunk_id": "1f2a3b4c5d6e7f80-ffeeddccbbaa00998877",
  "doc_id": "1f2a3b4c5d6e7f80",
  "url": "https://developers.weixin.qq.com/miniprogram/dev/reference/api/App.html",
  "title": "App",
  "nav_path": ["文档", "框架", "App"],
  "section_path": ["注册", "参数"],
  "chunk_type": "code",
  "chunk_text": "App({})",
  "related_code_ids": [],
  "related_text_ids": ["1f2a3b4c5d6e7f80-aabbccddeeff00112233"],
  "token_estimate": 3,
  "fetched_at": "2026-03-10T03:00:00+00:00"
}
```

### `outputs/failed.jsonl`

每行一个失败记录：

- `url`
- `stage`
- `error`
- `retry_count`
- `failed_at`

## 增量模式说明

增量状态保存在 `state/page_fingerprints.json`。

- `full` 模式会重新抓取全部发现页面，并覆盖写出 JSONL
- `incremental` 模式仍会访问当前发现到的页面，但只将指纹变化或新增页面写入 `framework_pages.jsonl` 和 `framework_chunks.jsonl`
- 指纹依据页面标题、正文文本和更新时间计算

这意味着增量模式的输出文件只包含“本次变化集”，不是历史全量快照。

## 处理规则

- URL 会做标准化、绝对化和去重
- 正文区域优先从 `main`、`.markdown-doc`、`.doc-content`、`.markdown-body`、`article` 中选择
- 切块按 `h1/h2/h3/h4` 生效
- 文本说明与代码示例会拆成独立 chunk，并通过 `related_code_ids` / `related_text_ids` 建立关联
- 空块会被过滤
- 超短块会与前一个块合并，降低噪音
- 抓取失败会按指数退避重试，默认 `3` 次

## 测试

运行测试：

```bash
python -m pytest crawler/tests -o cache_dir=state/.pytest_cache
```

当前测试覆盖：

- URL 标准化与去重
- HTML 正文与代码块提取
- 标题层级切块
- JSONL 字段完整性

## 已知限制

- 站点 DOM 结构依赖选择器，若微信文档页面改版，可能需要调整 [crawler/config.py](/d:/石卓/RAG_demo/crawler/config.py)
- 当前未实现“只请求疑似变更页再对比”的轻量增量发现，增量模式仍会重新访问所有发现页面
- 当前以中文页面为主，不处理多语言切换
- 在线 smoke test 需要本机网络可访问目标站点

## 关键文件

- [crawler/cli.py](/d:/石卓/RAG_demo/crawler/cli.py): CLI 入口
- [crawler/pipeline.py](/d:/石卓/RAG_demo/crawler/pipeline.py): 抓取主流程
- [crawler/discovery.py](/d:/石卓/RAG_demo/crawler/discovery.py): 侧边栏链接发现
- [crawler/extraction.py](/d:/石卓/RAG_demo/crawler/extraction.py): 页面内容抽取
- [crawler/chunking.py](/d:/石卓/RAG_demo/crawler/chunking.py): 标题层级切块
- [crawler/tests/test_extraction.py](/d:/石卓/RAG_demo/crawler/tests/test_extraction.py): 代表性提取测试
