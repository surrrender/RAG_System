from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(slots=True)
class OllamaGenerator:
    host: str
    model: str
    timeout: float

    def generate(self, prompt: str) -> str:
        text = "".join(self.generate_stream(prompt)).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text

    def generate_stream(self, prompt: str) -> Iterator[str]:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "ollama is required for local generation. Install the LLM project dependencies first."
            ) from exc

        client = ollama.Client(host=self.host, timeout=self.timeout)
        try:
            stream = client.generate(model=self.model, prompt=prompt, stream=True)
        except Exception as exc:  # pragma: no cover - exact exception type depends on ollama client
            raise RuntimeError(f"Failed to call Ollama at {self.host}: {exc}") from exc

        received_text = False
        try:
            for item in stream:
                text = str(item.get("response", ""))
                if not text:
                    continue
                received_text = True
                yield text
        except Exception as exc:  # pragma: no cover - exact exception type depends on ollama client
            raise RuntimeError(f"Failed to stream from Ollama at {self.host}: {exc}") from exc

        if not received_text:
            raise RuntimeError("Ollama returned an empty streaming response.")
