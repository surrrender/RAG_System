from typer.testing import CliRunner

from llm.cli import app
from llm.models import AnswerResult, RetrievedChunk


runner = CliRunner()


def test_ask_command_outputs_answer_and_citations(monkeypatch) -> None:
    def fake_answer_question(question: str, top_k: int | None = None) -> AnswerResult:
        assert question == "App 生命周期是什么？"
        assert top_k == 3
        return AnswerResult(
            question=question,
            answer="这是答案",
            citations=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    score=0.9,
                    title="App",
                    url="https://example.com/app",
                    section_path=["生命周期"],
                    text="App onLaunch",
                )
            ],
            model="llama3.1:8b",
            retrieval_count=1,
        )

    monkeypatch.setattr("llm.cli.answer_question", fake_answer_question)

    result = runner.invoke(app, ["ask", "App 生命周期是什么？", "--top-k", "3"])

    assert result.exit_code == 0
    assert "这是答案" in result.output
    assert "Citations:" in result.output


def test_serve_command_starts_uvicorn(monkeypatch) -> None:
    calls: list[tuple[object, str, int]] = []

    def fake_run(app_instance: object, host: str, port: int) -> None:
        calls.append((app_instance, host, port))

    monkeypatch.setattr("uvicorn.run", fake_run)

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    assert calls
