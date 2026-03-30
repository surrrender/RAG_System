from unittest.mock import patch

import pytest

from llm.generator import OllamaGenerator


def test_generator_returns_ollama_text() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float) -> None:
            assert host == "http://127.0.0.1:11434"
            assert timeout == 30.0

        def generate(self, model: str, prompt: str, stream: bool = False) -> list[dict[str, str | bool]]:
            assert model == "llama3.1:8b"
            assert prompt == "hello"
            assert stream is True
            return [{"response": "world", "done": True}]

    with patch("ollama.Client", FakeClient):
        assert generator.generate("hello") == "world"


def test_generator_raises_clear_error_on_failure() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float) -> None:
            pass

        def generate(self, model: str, prompt: str, stream: bool = False) -> list[dict[str, str | bool]]:
            raise RuntimeError("boom")

    with patch("ollama.Client", FakeClient), pytest.raises(RuntimeError, match="Failed to call Ollama"):
        generator.generate("hello")


def test_generator_stream_returns_chunked_text() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float) -> None:
            pass

        def generate(self, model: str, prompt: str, stream: bool = False) -> list[dict[str, str | bool]]:
            assert model == "llama3.1:8b"
            assert prompt == "hello"
            assert stream is True
            return [
                {"response": "line1\n", "done": False},
                {"response": "line2\n", "done": False},
                {"response": "line3", "done": True},
            ]

    with patch("ollama.Client", FakeClient):
        chunks = list(generator.generate_stream("hello"))

    assert "".join(chunks) == "line1\nline2\nline3"
    assert chunks


def test_generator_stream_raises_when_ollama_returns_no_text() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float) -> None:
            pass

        def generate(self, model: str, prompt: str, stream: bool = False) -> list[dict[str, str | bool]]:
            return [
                {"response": "", "done": False},
                {"response": "", "done": True},
            ]

    with patch("ollama.Client", FakeClient), pytest.raises(
        RuntimeError,
        match="empty streaming response",
    ):
        list(generator.generate_stream("hello"))
