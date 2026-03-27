# RAG System Workspace

这个仓库现在推荐使用一个顶层 Python 虚拟环境，而不是分别进入 `Crawler`、`Embedding_Indexing`、`LLM` 去切换各自的 `.venv`。

## 目录说明

- `Crawler`：文档爬取
- `Embedding_Indexing`：向量化、建索引、检索
- `LLM`：RAG 问答与 API
- `FrontEnd`：前端项目，保留自己的 Node.js 依赖管理

## 统一 Python 环境

在仓库根目录执行：

```bash
python3 scripts/bootstrap_python_workspace.py
source .venv/bin/activate
```

这个脚本会做三件事：

1. 在仓库根目录创建统一的 `.venv`
2. 扫描 `Crawler/.venv`、`Embedding_Indexing/.venv`、`LLM/.venv`
3. 尽量提取这些旧环境里已经安装过的第三方包版本，写入 `.workspace/constraints-from-subvenvs.txt`

随后脚本会把三个 Python 子项目都以 editable 方式安装到根目录 `.venv` 中：

```bash
-e ./Crawler
-e ./Embedding_Indexing
-e ./LLM
```

这意味着以后你不需要进入子目录重复切环境，只要在仓库根目录激活一次 `.venv` 即可。

## 为什么不用“直接复制旧 site-packages”

旧虚拟环境里的依赖虽然已经下载过，但直接复制 `site-packages` 风险很高：

- 可能夹带解释器路径
- 二进制依赖可能和目标环境不兼容
- editable 安装的链接关系容易失效

所以这里采用更稳妥的迁移方式：

- 优先复用旧环境中“已经装过的版本信息”
- 在新的顶层 `.venv` 中统一安装

这样比完全重装更接近你原来的依赖版本，同时后续维护也简单很多。

## 常用命令

激活根目录环境后，可以直接在仓库根目录运行：

```bash
crawler --help
embedding-indexing --help
python -m llm --help
```

如果你只想先生成约束文件，不立即安装：

```bash
python3 scripts/bootstrap_python_workspace.py --constraints-only
```

如果你想先创建 `.venv` 和约束文件，稍后再手动安装：

```bash
python3 scripts/bootstrap_python_workspace.py --skip-install
```

## 后续建议

- 逐步废弃子目录里的 `.venv`
- 把所有 Python 依赖统一安装到根目录 `.venv`
- `FrontEnd` 继续单独使用 `npm install`
