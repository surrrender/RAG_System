from pathlib import Path

from fastapi.testclient import TestClient

from llm.api import create_app
from llm.models import AnswerResult, Citation
from llm.storage import ConversationStore


class StubService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, list[dict[str, str]]]] = []
        self.stream_calls: list[tuple[str, int, list[dict[str, str]]]] = []
        self.warm_up_calls = 0

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def answer_question(self, question: str, top_k: int, history: list[object] | None = None) -> AnswerResult:
        normalized_history = [
            {"role": getattr(item, "role"), "content": getattr(item, "content")} for item in (history or [])
        ]
        self.calls.append((question, top_k, normalized_history))
        return AnswerResult(
            question=question,
            answer="answer",
            citations=[
                Citation(
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
                "data": {
                    "question": question,
                    "model": "llama3.1:8b",
                    "retrieval_count": 1,
                    "server_started_at_ms": 0.0,
                    "retrieval_finished_at_ms": 12.5,
                },
            },
            {"event": "delta", "data": {"text": "ans", "server_first_token_at_ms": 34.0}},
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
            {"event": "done", "data": {"answer": "answer", "server_completed_at_ms": 89.0}},
        ]


def _create_client(tmp_path: Path) -> tuple[TestClient, ConversationStore]:
    service = StubService()
    store = ConversationStore(tmp_path / "app.sqlite3")
    client = TestClient(create_app(service=service, store=store))
    client.app.state.stub_service = service
    return client, store


def test_app_warms_up_service_on_startup(tmp_path: Path) -> None:
    service = StubService()
    store = ConversationStore(tmp_path / "app.sqlite3")

    with TestClient(create_app(service=service, store=store)):
        assert service.warm_up_calls == 1


def test_conversation_crud_and_user_isolation(tmp_path: Path) -> None:
    client, _ = _create_client(tmp_path)

    created = client.post("/conversations", json={"user_id": "user-a"})
    assert created.status_code == 200
    conversation_id = created.json()["id"]

    listing = client.get("/conversations", params={"user_id": "user-a"})
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [conversation_id]

    isolated = client.get(f"/conversations/{conversation_id}/messages", params={"user_id": "user-b"})
    assert isolated.status_code == 404

    renamed = client.patch(
        f"/conversations/{conversation_id}",
        json={"user_id": "user-a", "title": "测试会话"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "测试会话"

    deleted = client.delete(f"/conversations/{conversation_id}", params={"user_id": "user-a"})
    assert deleted.status_code == 204
    assert client.get("/conversations", params={"user_id": "user-a"}).json() == []


def test_post_qa_persists_messages_and_uses_stored_history(tmp_path: Path) -> None:
    client, store = _create_client(tmp_path)
    service = client.app.state.stub_service
    conversation = store.create_conversation(user_id="user-a")
    store.add_message(
        user_id="user-a",
        conversation_id=conversation.id,
        role="user",
        content="先介绍 App",
        status="done",
    )
    store.add_message(
        user_id="user-a",
        conversation_id=conversation.id,
        role="assistant",
        content="App 是小程序入口。",
        status="done",
    )

    response = client.post(
        "/qa",
        json={
            "user_id": "user-a",
            "conversation_id": conversation.id,
            "question": "App 生命周期是什么？",
            "top_k": 4,
            "history": [{"role": "user", "content": "这段 history 不该被优先使用"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "answer"
    assert service.calls == [
        (
            "App 生命周期是什么？",
            4,
            [
                {"role": "user", "content": "先介绍 App"},
                {"role": "assistant", "content": "App 是小程序入口。"},
            ],
        )
    ]

    stored_messages = client.get(
        f"/conversations/{conversation.id}/messages",
        params={"user_id": "user-a"},
    ).json()
    assert [item["role"] for item in stored_messages] == ["user", "assistant", "user", "assistant"]
    assert stored_messages[-1]["content"] == "answer"


def test_post_qa_stream_returns_sse_events_and_persists_final_answer(tmp_path: Path) -> None:
    client, store = _create_client(tmp_path)
    service = client.app.state.stub_service
    conversation = store.create_conversation(user_id="user-a")
    store.add_message(
        user_id="user-a",
        conversation_id=conversation.id,
        role="user",
        content="先介绍 App",
        status="done",
    )

    response = client.post(
        "/qa/stream",
        json={
            "user_id": "user-a",
            "conversation_id": conversation.id,
            "question": "App 生命周期是什么？",
            "top_k": 4,
            "history": [{"role": "user", "content": "前端传来的历史不应覆盖数据库历史"}],
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

    stored_messages = client.get(
        f"/conversations/{conversation.id}/messages",
        params={"user_id": "user-a"},
    ).json()
    assert stored_messages[-1]["content"] == "answer"
    assert stored_messages[-1]["status"] == "done"
    assert stored_messages[-1]["model"] == "llama3.1:8b"
    assert stored_messages[-1]["retrieval_count"] == 1
