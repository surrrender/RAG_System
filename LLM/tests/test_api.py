from fastapi.testclient import TestClient

from llm.api import create_app
from llm.models import AnswerResult, RetrievedChunk


class StubService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def answer_question(self, question: str, top_k: int) -> AnswerResult:
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
