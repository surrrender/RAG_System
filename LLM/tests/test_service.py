from llm.generator import OllamaGenerator
from llm.models import ConversationTurn, RetrievedChunk
from llm.service import EMPTY_RESULT_ANSWER, QAService


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, int]] = []
        self.warm_up_calls = 0

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((question, top_k))
        return self._chunks


class StubGenerator:
    def __init__(self, text: str) -> None:
        self.model = "stub-model"
        self._text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._text

    def generate_stream(self, prompt: str) -> list[str]:
        self.prompts.append(prompt)
        midpoint = max(1, len(self._text) // 2)
        return [self._text[:midpoint], self._text[midpoint:]]


def test_service_returns_conservative_answer_when_no_chunks() -> None:
    service = QAService(retriever=StubRetriever([]), generator=StubGenerator("ignored"), max_context_chars=1000)

    result = service.answer_question("什么是 App？", top_k=5)

    assert result.answer == EMPTY_RESULT_ANSWER
    assert result.citations == []
    assert result.retrieval_count == 0


def test_service_warm_up_initializes_retriever() -> None:
    retriever = StubRetriever([])
    service = QAService(retriever=retriever, generator=StubGenerator("ignored"), max_context_chars=1000)

    service.warm_up()

    assert retriever.warm_up_calls == 1


def test_service_returns_answer_and_citations() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.8,
            title="App",
            url="https://example.com/app",
            section_path=["生命周期"],
            text="App 包含 onLaunch。",
        )
    ]
    retriever = StubRetriever(chunks)
    generator = StubGenerator("App 生命周期包括 onLaunch。")
    service = QAService(retriever=retriever, generator=generator, max_context_chars=1000)

    result = service.answer_question("App 生命周期是什么？", top_k=3)

    assert retriever.calls == [("App 生命周期是什么？", 3)]
    assert generator.prompts
    assert result.answer == "App 生命周期包括 onLaunch。"
    assert result.citations == chunks
    assert result.retrieval_count == 1


def test_service_streams_events_with_history() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.8,
            title="App",
            url="https://example.com/app",
            section_path=["生命周期"],
            text="App 包含 onLaunch。",
        )
    ]
    retriever = StubRetriever(chunks)
    generator = StubGenerator("App 生命周期包括 onLaunch。")
    service = QAService(retriever=retriever, generator=generator, max_context_chars=1000)

    events = list(
        service.stream_answer_question(
            "App 生命周期是什么？",
            top_k=3,
            history=[ConversationTurn(role="user", content="先解释一下 App")],
        )
    )

    assert retriever.calls == [("App 生命周期是什么？", 3)]
    assert events[0]["event"] == "meta"
    assert events[0]["data"]["server_started_at_ms"] == 0.0
    assert isinstance(events[0]["data"]["retrieval_finished_at_ms"], float)
    assert [event["event"] for event in events[1:3]] == ["delta", "delta"]
    assert isinstance(events[1]["data"]["server_first_token_at_ms"], float)
    assert "server_first_token_at_ms" not in events[2]["data"]
    assert events[-2]["event"] == "citations"
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["answer"] == "App 生命周期包括 onLaunch。"
    assert isinstance(events[-1]["data"]["server_completed_at_ms"], float)
    assert "先解释一下 App" in generator.prompts[0]


def test_service_streams_empty_result_without_citations() -> None:
    service = QAService(retriever=StubRetriever([]), generator=StubGenerator("ignored"), max_context_chars=1000)

    events = list(service.stream_answer_question("什么是 App？", top_k=5))

    assert [event["event"] for event in events] == ["meta", "delta", "citations", "done"]
    assert events[1]["data"]["text"] == EMPTY_RESULT_ANSWER
    assert isinstance(events[1]["data"]["server_first_token_at_ms"], float)
    assert isinstance(events[-1]["data"]["server_completed_at_ms"], float)
