from llm.models import RetrievedChunk
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
    assert "https://example.com/app" in prompt


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
