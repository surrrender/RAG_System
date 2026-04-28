# Frontend

独立的 React + Vite 前端，默认通过 `LLM` 服务的 `POST /qa/stream` 接口消费流式回答。

## 开发

安装依赖：

```bash
npm install
```

启动开发服务器：

```bash
npm run dev
```

默认会把 `/api/*` 代理到 `http://127.0.0.1:8000`。

如果需要自定义后端地址，可以设置：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 多会话行为

前端现在会在浏览器 `localStorage` 中自动生成并复用一个轻量 `user_id`，并基于后端接口实现：

- 会话列表
- 新建会话
- 切换历史会话
- 重命名会话
- 删除会话

发送问题时，前端只提交当前 `user_id`、`conversation_id`、`question` 与 `top_k`，历史消息由后端根据当前会话自动加载。

## 脚本

```bash
npm run dev
npm run build
npm run typecheck
npm run test
npm run test:e2e
npm run benchmark:latency
```

## 性能基准

前端会在页面内展示最近一次请求的流式时延指标；访问 `/?benchmark=1` 时，还会展示聚合结果。

运行浏览器级基准测试：

```bash
npm run benchmark:latency
```

默认会启动一个稳定的 mock SSE 服务和前端开发服务器，并把结果输出到：

```bash
benchmark-results/qa-latency.json
benchmark-results/qa-latency.csv
```

如果你已经启动了真实前端和后端，也可以从题目文件批量执行问答，并把每次的 5 个关键时延字段保存到一个结果文件：

```bash
npm run benchmark:file -- ./questions.txt ./benchmark-results/question-latency-results.json
```

这条命令会在完成采样后自动更新：

```bash
benchmark-results/question-latency-dashboard.html
```

默认读取规则：

- `.txt`：每行一个问题，空行和 `#` 注释会被忽略
- `.json`：字符串数组
- `.jsonl`：每行一个字符串，或 `{"question": "..."}` 对象

默认会连接到：

```bash
http://127.0.0.1:5173/?benchmark=1
```

如果你的前端地址不同，可以设置：

```bash
QA_BENCHMARK_APP_URL=http://127.0.0.1:4173/?benchmark=1 npm run benchmark:file -- ./questions.txt
```

输出文件中的每个元素只包含这 5 个字段，顺序与题目文件中的问题顺序一致：

- `time_to_first_visible_char_ms`
- `time_to_full_visible_answer_ms`
- `server_retrieval_ms`
- `server_time_to_first_token_ms`
- `server_total_ms`

## Benchmark 可视化

如果你已经拿到类似 `benchmark-results/question-latency-results.json` 这样的结果文件，可以先把它转换成 HTML 页面里可直接使用的 `rawData` 片段：

```bash
npm run benchmark:rawdata -- benchmark-results/question-latency-results.json
```

这条命令会默认直接更新：

```bash
benchmark-results/question-latency-dashboard.html
```

如果你想把结果写到单独文件里，方便后续复制到页面中：

```bash
npm run benchmark:rawdata -- benchmark-results/question-latency-results.json benchmark-results/raw-data-snippet.js
```

这个命令会输出：

```js
const rawData = [
  { time_to_first_visible_char_ms: 14478.1, time_to_full_visible_answer_ms: 27249, server_retrieval_ms: 5031.28 },
  ...
];
```

目前这个转换脚本只保留绘图需要的 3 个字段：

- `time_to_first_visible_char_ms`
- `time_to_full_visible_answer_ms`
- `server_retrieval_ms`

当前可直接打开的可视化页面示例在：

```bash
benchmark-results/question-latency-dashboard.html
```
