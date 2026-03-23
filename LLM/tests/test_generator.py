from unittest.mock import patch

import pytest

from llm.generator import OllamaGenerator


def test_generator_returns_ollama_text() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float) -> None:
            assert host == "http://127.0.0.1:11434"
            assert timeout == 30.0

        def generate(self, model: str, prompt: str) -> dict[str, str]:
            assert model == "llama3.1:8b"
            assert prompt == "hello"
            return {"response": "world"}

    with patch("ollama.Client", FakeClient):
        assert generator.generate("hello") == "world"


def test_generator_raises_clear_error_on_failure() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float) -> None:
            pass

        def generate(self, model: str, prompt: str) -> dict[str, str]:
            raise RuntimeError("boom")

    with patch("ollama.Client", FakeClient), pytest.raises(RuntimeError, match="Failed to call Ollama"):
        generator.generate("hello")
