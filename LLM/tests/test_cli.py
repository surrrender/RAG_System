import sys
from types import SimpleNamespace

from typer.testing import CliRunner

from llm.cli import app


runner = CliRunner()


class StubService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def stream_answer_question(self, question: str, top_k: int) -> list[dict[str, object]]:
        self.calls.append((question, top_k))
        return [
            {"event": "meta", "data": {"model": "llama3.1:8b", "retrieval_count": 1}},
            {"event": "delta", "data": {"text": "这是"}},
            {"event": "delta", "data": {"text": "答案"}},
            {
                "event": "citations",
                "data": {
                    "citations": [
                        {
                            "chunk_id": "chunk-1",
                            "score": 0.9,
                            "title": "App",
                            "url": "https://example.com/app",
                            "section_path": ["生命周期"],
                            "text": "App onLaunch",
                        }
                    ]
                },
            },
            {"event": "done", "data": {"answer": "这是答案"}},
        ]


def test_ask_command_streams_answer_and_citations(monkeypatch) -> None:
    service = StubService()
    monkeypatch.setattr("llm.cli.build_service", lambda settings: service)

    result = runner.invoke(app, ["ask", "App 生命周期是什么？", "--top-k", "3"])

    assert result.exit_code == 0
    assert service.calls == [("App 生命周期是什么？", 3)]
    assert "这是答案" in result.output
    assert "Citations:" in result.output


def test_serve_command_starts_uvicorn(monkeypatch) -> None:
    calls: list[tuple[object, str, int]] = []

    def fake_run(app_instance: object, host: str, port: int) -> None:
        calls.append((app_instance, host, port))

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    monkeypatch.setattr("llm.cli.create_app", lambda settings: object())

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    assert calls
