from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass

from llm.networking import is_local_service_url, normalize_local_service_url, protocol_hint


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
        # 下面这部分代码就是加载 ollama 客户端并调用生成接口的逻辑,包含了一些针对本地服务的特殊处理（比如绕过代理设置）
        normalized_host = normalize_local_service_url(self.host) or self.host
        local_host = is_local_service_url(normalized_host)
        try:
            with _local_service_proxy_guard(enabled=local_host):
                import ollama
        except ImportError as exc:
            raise RuntimeError(
                "ollama is required for local generation. Install the LLM project dependencies first."
            ) from exc

        client_kwargs = {"host": normalized_host, "timeout": self.timeout}
        if local_host:
            client_kwargs["trust_env"] = False
        with _local_service_proxy_guard(enabled=local_host):
            client = ollama.Client(**client_kwargs)
        try:
            stream = client.generate(model=self.model, prompt=prompt, stream=True, think=False)
        except Exception as exc:  # pragma: no cover - exact exception type depends on ollama client
            raise RuntimeError(
                f"Failed to call Ollama at {normalized_host}: {exc}. {protocol_hint(normalized_host, 'Ollama')}"
            ) from exc

        received_text = False
        try:
            for item in stream:
                text = str(item.get("response", ""))
                if not text:
                    continue
                received_text = True
                yield text
        except Exception as exc:  # pragma: no cover - exact exception type depends on ollama client
            raise RuntimeError(
                f"Failed to stream from Ollama at {normalized_host}: {exc}. {protocol_hint(normalized_host, 'Ollama')}"
            ) from exc

        if not received_text:
            raise RuntimeError("Ollama returned an empty streaming response.")


@contextmanager
def _local_service_proxy_guard(enabled: bool):
    if not enabled:
        yield
        return

    proxy_keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    snapshot = {key: os.environ.pop(key, None) for key in proxy_keys}
    try:
        yield
    finally:
        for key, value in snapshot.items():
            if value is not None:
                os.environ[key] = value
