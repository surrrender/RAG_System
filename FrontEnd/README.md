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

## 脚本

```bash
npm run dev
npm run build
npm run typecheck
npm run test
```
