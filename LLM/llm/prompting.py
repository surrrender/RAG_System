from __future__ import annotations

from llm.models import ConversationTurn, RetrievedChunk


SYSTEM_PROMPT_OLD = """你是一个面向微信小程序文档的问答助手。
请严格依据提供的检索资料回答，不要编造资料中没有的信息。
如果资料不足以支持明确答案，请直接说明“未找到足够依据”，并简要指出还缺什么信息。
回答要精准、丰富，优先给出结论，再给出必要说明。"""

SYSTEM_PROMPT = """
你是一个专业的跨端容器开发助手。

你必须严格依据检索资料回答问题。
禁止使用外部知识、训练记忆或自行推测。

回答目标：
- 准确
- 可验证
- 对开发者有帮助
- 避免幻觉
- 明确限制条件

回答流程：

1. 理解用户问题
- 提取涉及的 API、组件、配置、生命周期或错误现象
- 识别是否存在平台、版本或场景限制

2. 筛选证据
- 优先选择直接相关的检索内容
- 忽略弱相关信息
- 如果多个资料存在冲突或适用条件不同，需要明确说明

3. 基于资料推理
- 结合多个资料逐步分析
- 不允许超出资料进行推断
- 不允许补全文档中不存在的行为

4. 输出最终答案
- 先给结论
- 再给依据和说明
- 涉及代码时使用 ```language
- 代码必须与资料一致

5. 资料不足时
- 明确说明“未找到足够依据”
- 指出缺失的信息或建议补充方向

特别注意：
- 不要虚构 API
- 不要虚构参数
- 不要虚构返回值
- 不要猜测框架行为
- 必须指出版本/平台限制
- 必须指出前置条件
"""

MAX_HISTORY_TURNS = 4


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    max_context_chars: int,
    history: list[ConversationTurn] | None = None,
) -> str:
    context_blocks = []
    remaining = max_context_chars

    for index, chunk in enumerate(chunks, start=1):
        block = _format_chunk(index=index, chunk=chunk)
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        if block:
            context_blocks.append(block)
            remaining -= len(block)

    joined_context = "\n\n".join(context_blocks).strip()
    history_block = _format_history(history or [])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{history_block}"
        f"用户问题：\n{question.strip()}\n\n"
        f"检索资料：\n{joined_context or '无可用资料'}\n\n"
        "代码格式:如果输出内容中包含代码,请使用``包含单行代码或者```来包含代码片段"
        "请输出答案，并只使用检索资料能支持的内容。"
    )


def _format_chunk(index: int, chunk: RetrievedChunk) -> str:
    section_path = " > ".join(item.strip() for item in (chunk.section_path or []) if item.strip())
    content = _trim_chunk_content((chunk.text or "").strip())
    lines = [
        f"[资料 {index}]",
        f"标题：{chunk.title or '未知标题'}",
        f"章节：{section_path or '未知章节'}",
        f"内容：{content}",
    ]
    return "\n".join(lines).strip()


def _format_history(history: list[ConversationTurn]) -> str:
    normalized = [
        turn for turn in history if turn.role in {"user", "assistant"} and turn.content.strip()
    ][-(MAX_HISTORY_TURNS * 2):]
    if not normalized:
        return ""

    lines = ["对话历史："]
    for turn in normalized:
        speaker = "用户" if turn.role == "user" else "助手"
        lines.append(f"[{speaker}] {turn.content.strip()}")

    return "\n".join(lines) + "\n\n"


def _trim_chunk_content(content: str, max_chars: int = 1200) -> str:
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 1].rstrip() + "…"
