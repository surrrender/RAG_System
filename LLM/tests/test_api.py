from fastapi.testclient import TestClient

from llm.api import create_app
from llm.models import AnswerResult, RetrievedChunk


class StubService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.stream_calls: list[tuple[str, int, list[dict[str, str]]]] = []

    def answer_question(self, question: str, top_k: int, history: list[object] | None = None) -> AnswerResult:
        self.calls.append((question, top_k))
        return AnswerResult(
            question=question,
            answer="answer",
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

    def stream_answer_question(
        self, question: str, top_k: int, history: list[object] | None = None
    ) -> list[dict[str, object]]:
        normalized_history = [
            {"role": getattr(item, "role"), "content": getattr(item, "content")} for item in (history or [])
        ]
        self.stream_calls.append((question, top_k, normalized_history))
        return [
            {
                "event": "meta",
                "data": {"question": question, "model": "llama3.1:8b", "retrieval_count": 1},
            },
            {"event": "delta", "data": {"text": "ans"}},
            {"event": "delta", "data": {"text": "wer"}},
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
            {"event": "done", "data": {"answer": "answer"}},
        ]


def test_post_qa_returns_answer() -> None:
    service = StubService()
    client = TestClient(create_app(service=service))

    response = client.post("/qa", json={"question": "App 生命周期是什么？", "top_k": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer"
    assert body["retrieval_count"] == 1
    assert body["citations"][0]["chunk_id"] == "chunk-1"
    assert service.calls == [("App 生命周期是什么？", 4)]


def test_post_qa_rejects_empty_question() -> None:
    client = TestClient(create_app(service=StubService()))

    response = client.post("/qa", json={"question": "", "top_k": 4})

    assert response.status_code == 422


def test_post_qa_stream_returns_sse_events() -> None:
    service = StubService()
    client = TestClient(create_app(service=service))

    response = client.post(
        "/qa/stream",
        json={
            "question": "App 生命周期是什么？",
            "top_k": 4,
            "history": [{"role": "user", "content": "先介绍 App"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: meta" in response.text
    assert "event: delta" in response.text
    assert "event: citations" in response.text
    assert "event: done" in response.text
    assert service.stream_calls == [
        ("App 生命周期是什么？", 4, [{"role": "user", "content": "先介绍 App"}])
    ]
