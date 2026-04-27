from llm.networking import normalize_local_service_url, protocol_hint


def test_normalize_local_https_url_to_http() -> None:
    assert normalize_local_service_url("https://127.0.0.1:6333") == "http://127.0.0.1:6333"
    assert normalize_local_service_url("https://localhost:11434") == "http://localhost:11434"


def test_keep_remote_https_url_unchanged() -> None:
    assert normalize_local_service_url("https://cloud.qdrant.io") == "https://cloud.qdrant.io"


def test_protocol_hint_mentions_http_for_local_urls() -> None:
    hint = protocol_hint("https://127.0.0.1:6333", "Qdrant")
    assert "http://" in hint
    assert "Qdrant" in hint
