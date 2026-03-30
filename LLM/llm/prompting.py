from __future__ import annotations

from llm.models import ConversationTurn, RetrievedChunk


SYSTEM_PROMPT = """你是一个面向微信小程序文档的问答助手。
请严格依据提供的检索资料回答，不要编造资料中没有的信息。
如果资料不足以支持明确答案，请直接说明“未找到足够依据”，并简要指出还缺什么信息。
回答要精准、丰富，优先给出结论，再给出必要说明。"""


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
    section_path = " > ".join(chunk.section_path or [])
    lines = [
        f"[资料 {index}]",
        f"标题：{chunk.title or '未知标题'}",
        f"章节：{section_path or '未知章节'}",
        f"链接：{chunk.url or '未知链接'}",
        f"相似度：{chunk.score:.4f}",
        f"内容：{(chunk.text or '').strip()}",
    ]
    return "\n".join(lines).strip()


def _format_history(history: list[ConversationTurn]) -> str:
    normalized = [
        turn for turn in history if turn.role in {"user", "assistant"} and turn.content.strip()
    ][-6:]
    if not normalized:
        return ""

    lines = ["对话历史："]
    for turn in normalized:
        speaker = "用户" if turn.role == "user" else "助手"
        lines.append(f"[{speaker}] {turn.content.strip()}")

    return "\n".join(lines) + "\n\n"
