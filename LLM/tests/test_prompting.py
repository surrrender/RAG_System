from llm.models import ConversationTurn, RetrievedChunk
from llm.prompting import build_prompt


def test_build_prompt_includes_citations() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.9,
            title="App",
            url="https://example.com/app",
            section_path=["生命周期", "onLaunch"],
            text="App 会在小程序初始化时触发 onLaunch。",
        )
    ]

    prompt = build_prompt("App 生命周期是什么？", chunks, max_context_chars=1000)

    assert "用户问题" in prompt
    assert "App" in prompt
    assert "生命周期 > onLaunch" in prompt
    assert "App 会在小程序初始化时触发 onLaunch。" in prompt
    assert "https://example.com/app" not in prompt
    assert "相似度" not in prompt


def test_build_prompt_truncates_context() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.9,
            title="Long",
            url="https://example.com/long",
            section_path=["A"],
            text="x" * 500,
        )
    ]

    prompt = build_prompt("test", chunks, max_context_chars=120)

    assert len(prompt) < 600


def test_build_prompt_trims_chunk_content() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.9,
            title="Long",
            url="https://example.com/long",
            section_path=["A"],
            text="x" * 500,
        )
    ]

    prompt = build_prompt("test", chunks, max_context_chars=1000)

    assert "内容：" in prompt
    assert "x" * 400 not in prompt
    assert "…" in prompt


def test_build_prompt_includes_recent_history() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.9,
            title="App",
            url="https://example.com/app",
            section_path=["生命周期", "onLaunch"],
            text="App 会在小程序初始化时触发 onLaunch。",
        )
    ]

    prompt = build_prompt(
        "App 生命周期是什么？",
        chunks,
        max_context_chars=1000,
        history=[
            ConversationTurn(role="user", content="先介绍一下 App"),
            ConversationTurn(role="assistant", content="App 用于注册小程序。"),
        ],
    )

    assert "对话历史" in prompt
    assert "[用户] 先介绍一下 App" in prompt
    assert "[助手] App 用于注册小程序。" in prompt


def test_build_prompt_keeps_last_four_history_turns() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.9,
            title="App",
            url="https://example.com/app",
            section_path=["生命周期"],
            text="App 会在小程序初始化时触发 onLaunch。",
        )
    ]

    history = [
        ConversationTurn(role="user", content="u1"),
        ConversationTurn(role="assistant", content="a1"),
        ConversationTurn(role="user", content="u2"),
        ConversationTurn(role="assistant", content="a2"),
        ConversationTurn(role="user", content="u3"),
        ConversationTurn(role="assistant", content="a3"),
        ConversationTurn(role="user", content="u4"),
        ConversationTurn(role="assistant", content="a4"),
        ConversationTurn(role="user", content="u5"),
        ConversationTurn(role="assistant", content="a5"),
    ]

    prompt = build_prompt("App 生命周期是什么？", chunks, max_context_chars=1000, history=history)

    assert "[用户] u1" not in prompt
    assert "[助手] a1" not in prompt
    assert "[用户] u2" in prompt
    assert "[助手] a5" in prompt
