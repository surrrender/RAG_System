from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OllamaGenerator:
    host: str
    model: str
    timeout: float

    def generate(self, prompt: str) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "ollama is required for local generation. Install the LLM project dependencies first."
            ) from exc

        client = ollama.Client(host=self.host, timeout=self.timeout)
        try:
            response = client.generate(model=self.model, prompt=prompt)
        except Exception as exc:  # pragma: no cover - exact exception type depends on ollama client
            raise RuntimeError(f"Failed to call Ollama at {self.host}: {exc}") from exc

        text = str(response.get("response", "")).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text
