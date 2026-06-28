import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm.generator import OllamaGenerator


def _patch_ollama_client(fake_client: type[object]):
    return patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=fake_client)})


def test_generator_raises_clear_error_on_failure() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float, **kwargs: object) -> None:
            assert host == "http://127.0.0.1:11434"
            assert timeout == 30.0

        def generate(self, model: str, prompt: str, stream: bool = False, **kwargs: object):
            raise RuntimeError("boom")

    with _patch_ollama_client(FakeClient), pytest.raises(RuntimeError, match="Failed to call Ollama"):
        list(generator.generate_stream("hello"))


def test_generator_stream_returns_chunked_text() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float, **kwargs: object) -> None:
            assert host == "http://127.0.0.1:11434"
            assert timeout == 30.0

        def generate(self, model: str, prompt: str, stream: bool = False, **kwargs: object):
            assert model == "llama3.1:8b"
            assert prompt == "hello"
            assert stream is True
            return [
                {"response": "line1\n", "done": False},
                {"response": "line2\n", "done": False},
                {"response": "line3", "done": True},
            ]

    with _patch_ollama_client(FakeClient):
        chunks = list(generator.generate_stream("hello"))

    assert "".join(chunks) == "line1\nline2\nline3"
    assert chunks


def test_generator_stream_raises_when_ollama_returns_no_text() -> None:
    generator = OllamaGenerator(host="http://127.0.0.1:11434", model="llama3.1:8b", timeout=30.0)

    class FakeClient:
        def __init__(self, host: str, timeout: float, **kwargs: object) -> None:
            assert host == "http://127.0.0.1:11434"
            assert timeout == 30.0

        def generate(self, model: str, prompt: str, stream: bool = False, **kwargs: object):
            return [
                {"response": "", "done": False},
                {"response": "", "done": True},
            ]

    with _patch_ollama_client(FakeClient), pytest.raises(
        RuntimeError,
        match="empty streaming response",
    ):
        list(generator.generate_stream("hello"))
